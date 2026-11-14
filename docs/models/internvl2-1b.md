# InternVL2-1B

## 模型概况

`OpenGVLab/InternVL2-1B` 是 InternVL 2.0 系列中最小的视觉语言模型：

| 组件 | 规格 |
| --- | --- |
| 视觉塔 | InternViT-300M-448px（24 层，patch 14，输入 448×448） |
| 语言塔 | Qwen2-0.5B-Instruct（24 层，hidden 896，2 个 KV 头，head_dim 64） |
| 架构名 | `InternVLChatModel`（vLLM 有原生实现，但模型仓库 `config.json` 带 `auto_map`，**必须加 `--trust-remote-code`**，见「排错」第一条） |
| 模态 | 文本 + 图像（**不支持视频**） |

**图像如何变成 token**：图片按 `max_dynamic_patch`（默认 12）切成若干 448×448 的块，另外附带 1 张整体缩略图；每块 448/14=32 → 32×32=1024 个 patch，经 `downsample_ratio=0.5` 的像素重排压缩到 **256 token**。因此单图最多 13 块 ≈ **3328 个视觉 token**；一张普通方形照片（1024×1024）通常为 2×2=4 块 + 缩略图，约 1.3k token。这些 token 会替换 prompt 中的 `<image>` 占位符，再与文本 token 一起送进语言塔。

## 快速开始

### 离线推理

```bash
python examples/basic/offline_inference/internvl2_1b.py \
  --model OpenGVLab/InternVL2-1B \
  --image ./demo.jpg
```

### OpenAI 兼容服务

```bash
vllm serve OpenGVLab/InternVL2-1B \
  --served-model-name internvl2-1b \
  --max-model-len 8192 \
  --limit-mm-per-prompt image=1 \
  --gpu-memory-utilization 0.9 \
  --trust-remote-code
```

> `--trust-remote-code` 不可省略：模型的 `config.json` 里 `auto_map` 指向仓库自带的
> `configuration_internvl_chat.py` / `modeling_internvl_chat.py`，vLLM 在构造
> `EngineArgs` 时会先调用 transformers 的 `AutoConfig.from_pretrained`，
> 未授权远程代码会直接抛 `ValueError: ... contains custom code which must be executed ...`。

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
resp = client.chat.completions.create(
    model="internvl2-1b",
    messages=[{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        {"type": "text", "text": "这张图里有什么？"},
    ]}],
    max_tokens=128,
)
print(resp.choices[0].message.content)
```

## 参数说明

| 参数 | 建议值 | 说明 |
| --- | --- | --- |
| `--max-model-len` | 8192 | 视觉 token 上限约 3.3k，需留出文本与生成长度。注意本模型 KV 极省（约 12KB/token，8192 上下文 ≈ 0.1GB），**下调它主要是控制 token 预算，而非省显存** |
| `--limit-mm-per-prompt` | `image=1` | 限制每 prompt 图片数，避免视觉 token 超限 |
| `--gpu-memory-utilization` | 0.9（8GB 卡）/ 0.85 | 权重仅约 1.7GB（bf16），主要余量留给视觉激活与 CUDA graph |
| `--trust-remote-code` | **必填** | 模型 `config.json` 含 `auto_map`，不加会在 AutoConfig 阶段报错退出 |

显存参考：bf16 权重 ≈ 1.7GB，完整加载后 8GB 显卡可稳定运行 8k 上下文；显存 ≤6GB 时建议 `--max-model-len 4096 --enforce-eager`。

## 限制

- **不支持视频输入**：vLLM 中视频能力仅 InternVL2.5 / InternVL3 / InternVL3.5 具备。
- 极端长宽比图片（如长截图）会吃满 12+1 块，接近 3.3k 视觉 token；`max-model-len` 设得过小时会报 `prompt too long`。
- **环境要求：CUDA 12.9 或 13.0**（对应 `cu129` / `cu130` 轮子）。本仓 `requirements/cuda.txt` 钉 `torch==2.14.0`，`setup.py` 只映射这两个 CUDA 主版本；CUDA 12.1 / 12.4 等更老环境无法匹配。
- RTX 50 系（Blackwell，sm_120）自 CUDA 12.8 起受支持，故上述两种变体均可覆盖；更老的 torch 轮子会报 `no kernel image is available for execution on the device`。

## 排错

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| `ValueError: The repository for ... contains custom code which must be executed to correctly load the model` / `Please pass the argument 'trust_remote_code=True'` | 模型仓库带 `auto_map` 自定义代码，`AutoConfig` 被安全机制拦截（实测 vLLM 0.32.0 必现） | 启动命令加 `--trust-remote-code`（离线 API 加 `trust_remote_code=True`） |
| `No available memory for cache blocks` | 显存被其他进程占用或利用率设太高 | 降 `--gpu-memory-utilization`，或 `--max-model-len 4096` |
| 输出里 `<image>` 原样出现 | 图像未被 processor 替换 | 确认 `--limit-mm-per-prompt image=1`，且请求为 1 图 + 1 文本 |
| `number of image tokens mismatch` | 切块数与占位符数不一致 | 检查 `max_dynamic_patch`，客户端压图长边 ≤1024 |
| 首 token 很慢 | 首次 CUDA graph 捕获 | 正常现象；调试可加 `--enforce-eager` |
| `no kernel image is available` | GPU 架构与 torch 轮子不匹配 | 换成 `cu129` / `cu130` 轮子（CUDA ≥ 12.9），或换卡 |

## 实测记录

| 项 | 值 |
| --- | --- |
| 硬件 | NVIDIA GeForce RTX 4090D 24GB（24564 MiB），驱动 580.76.05，`CUDA Version: 13.0` |
| 软件 | CUDA 13.0 / torch 2.12.1+cu130 / vLLM 0.32.0 / transformers 5.19.0 / Python 3.12（Ubuntu 22.04） |
| 模型体积 | 20 个文件，`du -sh` = 1.9 GB |
| 是否需 `--trust-remote-code` | **是**（不加必报 `ValueError: ... contains custom code ...`） |
| 加载耗时 | `Loading weights took 0.54 seconds`，`Model loading took 1.76 GiB memory and 3.5 seconds` |
| 显存占用 | 权重 2.07 GiB；`--gpu-memory-utilization 0.85` → `Available KV cache memory: 17.89 GiB` |
| KV 容量 | `GPU KV cache size: 1,511,222 tokens`；`Maximum concurrency for 8,192 tokens per request: 184.48x` |
| CUDA graph | 51/51 捕获完成，耗时 2 s，额外 0.25 GiB |
| 注意力后端 | `Using FlashAttention version 2 as backend` |
| 端到端启动 | 约 5 分钟（含 flashinfer 采样 JIT 首次编译、torch.compile dynamo 7.8 s） |
| 单次图文推理 | 纯色图判色 → `Orange`；蓝底红圆 → `The image consists of a simple, solid color background with a central red circle. The shape in the image is a circle.` |
