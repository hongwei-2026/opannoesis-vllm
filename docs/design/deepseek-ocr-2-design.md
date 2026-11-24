# [Design] #7 DeepSeek-OCR-2 适配方案

## 目标
在 OpenNoesis/vllm 仓库中验证 DeepSeek-OCR-2 可被 vLLM 加载并完成推理：
能启动 serving、发出至少一次正确推理请求、留下可复现步骤。

## 实现路径
1. 模型权重：从 ModelScope/HuggingFace 下载 DeepSeek-OCR-2 权重
   （约 3B 参数，推理需 12GB 以上显存，计划使用昇腾 Notebook 环境）
2. 适配验证：使用 vLLM 加载模型，确认模型架构被 vLLM 正确识别；
   如遇不兼容，定位并修改模型注册相关路径（vllm/model_executor/models/），
   注册 DeepSeek-OCR-2 的模型实现。
3. 启动 serving：`vllm serve <模型路径> --port 8000`，
   以 OpenAI 兼容接口对外提供服务。
4. 推理验证：用 curl / Python 客户端向 `/v1/chat/completions`
   发送一张文档图片请求，确认返回正确的 OCR 识别结果。

## 交付物
- 可复现的启动命令与简要使用说明（docs）
- 至少一次成功推理的运行记录（截图/日志）
- 实现 PR 关联 Issue #7（正文写 Fixes #7）

## 验收对照
- [ ] 本仓环境加载 DeepSeek-OCR-2 并完成至少一次推理
- [ ] 有可复现的启动命令与简短使用说明
- [ ] 实现 PR 关联本 Issue 并通过维护者审核合入