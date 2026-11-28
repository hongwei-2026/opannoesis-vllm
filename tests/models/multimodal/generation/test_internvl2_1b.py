# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""InternVL2-1B 多模态回归用例（Issue #8）。

说明：
- 标记为 core_model，需要 GPU；按仓库惯例可归入 nightly（1B 模型显存占用小，
  权重仅约 1.76 GiB，单卡 8GB 即可跑）。
- 断言刻意保持宽松（非空 + 无未替换占位符），避免因采样波动导致 flaky；
  同时用 enforce_eager=True 降低 CI 环境 CUDA graph 相关的不稳定。
- 必须传 trust_remote_code=True：该模型 config.json 带 auto_map（指向仓库内
  configuration_internvl_chat.py / modeling_internvl_chat.py），不传会在
  transformers AutoConfig 阶段抛
  ValueError: ... contains custom code which must be executed ...
- 图片以 base64 data URL（字符串）形式传入 chat 接口：LLM.chat 会先把 messages
  交给 pydantic 校验 ChatCompletionMessageParam，其中 image_url.url 的类型是
  str；直接塞 PIL.Image 会抛 ValidationError（单图用例曾因此在 CI 失败）。
"""

import io

import pybase64
import pytest
from PIL import Image

from vllm import LLM, SamplingParams

MODEL_NAME = "OpenGVLab/InternVL2-1B"
PROMPT = "What is the content of this image? Answer in one sentence."


def _image(size: tuple[int, int] = (512, 512)) -> Image.Image:
    """构造一张确定性的测试图（橙色底 + 白色棋盘块），避免依赖外部资源。"""
    img = Image.new("RGB", size, (240, 130, 40))
    for x in range(64, size[0] - 64, 64):
        for y in range(64, size[1] - 64, 64):
            if (x // 64 + y // 64) % 2 == 0:
                img.paste((255, 255, 255), (x, y, x + 32, y + 32))
    return img


def _image_part(img: Image.Image) -> dict:
    """把 PIL 图编码为 chat 接口要求的 data URL 字符串。"""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    encoded = pybase64.b64encode(buf.getvalue()).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{encoded}"},
    }


@pytest.fixture(scope="module")
def internvl2_1b():
    return LLM(
        model=MODEL_NAME,
        dtype="bfloat16",
        max_model_len=8192,
        limit_mm_per_prompt={"image": 1},
        enforce_eager=True,
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
    )


@pytest.mark.core_model
def test_internvl2_1b_single_image(internvl2_1b):
    """单图问答：能加载、能出非空回答，且 <image> 占位符已被替换。"""
    conversation = [
        {
            "role": "user",
            "content": [
                _image_part(_image()),
                {"type": "text", "text": PROMPT},
            ],
        }
    ]

    outputs = internvl2_1b.chat(
        [conversation],
        sampling_params=SamplingParams(temperature=0.0, max_tokens=64),
    )

    text = outputs[0].outputs[0].text.strip()
    assert text, "模型返回空字符串"
    assert "<image>" not in text


@pytest.mark.core_model
def test_internvl2_1b_text_only(internvl2_1b):
    """纯文本路径不应被多模态改动破坏。"""
    outputs = internvl2_1b.generate(
        ["Hello, my name is"],
        sampling_params=SamplingParams(temperature=0.0, max_tokens=16),
    )
    assert outputs[0].outputs[0].text.strip()


@pytest.mark.core_model
def test_internvl2_1b_mm_limit_exceeded(internvl2_1b):
    """limit_mm_per_prompt={"image": 1} 时，单请求传 2 张图应被拒绝。"""
    conversation = [
        {
            "role": "user",
            "content": [
                _image_part(_image()),
                _image_part(_image()),
                {"type": "text", "text": PROMPT},
            ],
        }
    ]

    with pytest.raises(ValueError):
        internvl2_1b.chat(
            [conversation],
            sampling_params=SamplingParams(temperature=0.0, max_tokens=16),
        )
