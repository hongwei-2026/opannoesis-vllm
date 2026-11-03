<!-- SPDX-License-Identifier: Apache-2.0 -->
# Llama-3.2-3B-Instruct 部署与验证指南

本文档记录 `meta-llama/Llama-3.2-3B-Instruct` 在本仓 vLLM 环境下的适配验证结果，
包含可复现的下载、启动与推理步骤（已在华为昇腾 910B2 + CANN 9.0.0 环境实测通过）。

## 环境要求

- Python 3.10+
- vLLM（本仓）+ vllm-ascend 0.21.0rc1 / torch-npu 2.10.0（昇腾环境）
- 磁盘空间 >= 10 GB（模型权重约 6.5 GB）

## 1. 下载模型

模型权重可从 ModelScope 镜像获取（国内网络无需 HuggingFace 授权）：

```bash
python3 -c "from modelscope import snapshot_download; \
  snapshot_download('LLM-Research/Llama-3.2-3B-Instruct', cache_dir='/root/models', \
  allow_patterns=['*.json','*.safetensors','tokenizer*'])"
```

> 提示：`allow_patterns` 用于跳过 `original/consolidated.00.pth`（约 6 GB 的原始权重），vLLM 只需要 safetensors 分片。

## 2. 启动推理服务

```bash
# 昇腾环境需先初始化 CANN 环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh

vllm serve /root/models/LLM-Research/Llama-3___2-3B-Instruct \
  --served-model-name llama-3.2-3b-instruct \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 4096
```

看到 `Application startup complete.` 即启动成功。

> 说明：本仓配合 vllm-ascend 插件使用，插件会自动将 NPU 注册为推理平台，
> 因此 `vllm serve` 无需额外的 `--device npu` 参数（手动指定旧式设备参数反而可能报错）。

> 注意：容器环境挂载 NPU 后，设备编号从 0 开始。
> 请勿设置 `ASCEND_RT_VISIBLE_DEVICES` 为宿主机卡号（如 6），
> 否则会报 `aclInit error 107001 Invalid device ID`。

## 3. 发送推理请求

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"llama-3.2-3b-instruct","messages":[{"role":"user","content":"Introduce yourself in one sentence."}],"max_tokens":100}'
```

实测返回示例：

```json
{
  "model": "llama-3.2-3b-instruct",
  "choices": [{
    "message": {"role": "assistant",
      "content": "我是一个人工智能助手，能够提供信息、回答问题和完成任务，致力于帮助您找到所需的资源和解决问题。"},
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 41, "completion_tokens": 34, "total_tokens": 75}
}
```

## 4. 回归测试

```bash
pytest tests/basic_correctness/test_llama_3_2_3b_instruct.py -v
```

该用例通过 vLLM 离线推理接口加载模型并校验输出非空，验证模型可被正常加载与推理。

## 验证结论

| 项目 | 结果 |
| --- | --- |
| 模型加载 | 通过 |
| 服务启动（OpenAI 兼容接口） | 通过 |
| 推理请求返回 | 通过（finish_reason=stop） |
| 环境 | 昇腾 910B2 / CANN 9.0.0 / vllm-ascend 0.21.0rc1 |
