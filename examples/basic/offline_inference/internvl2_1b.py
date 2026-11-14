# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""InternVL2-1B 离线推理最小示例。

用法（需先安装本仓 vllm）：
    python examples/basic/offline_inference/internvl2_1b.py \
        --model OpenGVLab/InternVL2-1B --image ./demo.jpg
"""

import argparse
import io

import pybase64
from PIL import Image

from vllm import LLM, SamplingParams

# 注意：这里用的是 InternVL 官方 prompt 包装，<image> 占位符由 processor 替换
DEFAULT_QUESTION = "<image>\n这张图里有什么？请用一句中文描述。"


def _image_to_data_url(image: Image.Image) -> str:
    """把 PIL 图编码成 data URL 字符串。

    LLM.chat 会先用 pydantic 校验 ChatCompletionMessageParam，其中
    image_url.url 的类型是 str；直接传 PIL.Image 会抛 ValidationError。
    """
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    encoded = pybase64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def run_internvl2_1b(
    questions: list[str],
    modality: str,
    model: str = "OpenGVLab/InternVL2-1B",
    image_path: str | None = None,
) -> list[str]:
    """InternVL2-1B 的最小可复现路径：单图 + 单问，返回生成文本。"""
    assert modality == "image"
    if image_path is None:
        raise ValueError("InternVL2-1B 示例需要提供 --image")

    # 压缩大图，避免动态切块把视觉 token 拉到上限（13 块 ≈ 3.3k token）
    image = Image.open(image_path).convert("RGB")
    image.thumbnail((1024, 1024))

    llm = LLM(
        model=model,
        max_model_len=8192,
        limit_mm_per_prompt={"image": 1},
        gpu_memory_utilization=0.9,
        dtype="bfloat16",
        # InternVL2 的 config.json 带 auto_map（指向仓库内 modeling_*.py），
        # vLLM 初始化时会先走 transformers AutoConfig，不加此参数会抛
        # ValueError: ... contains custom code which must be executed ...
        trust_remote_code=True,
    )

    # 离线 chat 接口的 image_url.url 需为字符串，这里传已缩放图片的 data URL，
    # 保证上面 thumbnail 的尺寸真正生效
    data_url = _image_to_data_url(image)
    conversations = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": q.replace("<image>\n", "")},
                ],
            }
        ]
        for q in questions
    ]

    outputs = llm.chat(
        conversations, sampling_params=SamplingParams(temperature=0.2, max_tokens=128)
    )

    results = []
    for q, out in zip(questions, outputs):
        text = out.outputs[0].text
        print(f"Prompt: {q!r}\n{'=' * 60}\n{text}\n")
        results.append(text)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="InternVL2-1B offline inference")
    parser.add_argument("--model", default="OpenGVLab/InternVL2-1B")
    parser.add_argument("--image", required=True, help="本地图片路径")
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    args = parser.parse_args()

    run_internvl2_1b([args.question], "image", args.model, args.image)


if __name__ == "__main__":
    main()
