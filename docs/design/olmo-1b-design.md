# [Design] #11 OLMo-1B 昇腾适配方案

## 目标

在 OpenNoesis/vllm 仓库中，把 `allenai/OLMo-1B-hf` 在**昇腾 NPU** 上跑通：
能启动 serving、发出至少一次正确推理请求、留下可复现步骤。

## 背景与现状

| 检查项 | 结果 |
| --- | --- |
| GPU 侧是否已支持 | 是。`docs/models/supported_models.md` 已登记 `OlmoForCausalLM`，走 Transformers modeling backend |
| 昇腾侧是否已支持 | **否**。vllm-ascend 支持列表中没有 OLMo（只有 Molmo，属多模态） |
| 本仓性质 | GPU 版 vLLM，无 NPU 依赖；昇腾能力由 `vllm-ascend` 插件提供 |
| 本地硬件 | RTX 3060（无 NPU），NPU 验证在 CANNLab 进行 |

结论：本任务的"适配"指**昇腾侧**。GPU 上能跑不算完成，需要证明 NPU 上能跑。

## 环境

- **算力**：GitCode CANNLab，1×NPU 910B3 / 16 vCPU / 32 GiB，免费 100 小时，WebIDE 或 VS Code 连接
- **版本**：vllm-ascend 与 vLLM 必须同版本配套，CANN / torch-npu 按官方验证矩阵整组选择。
  查询时官方 latest 的验证组合为 vLLM `0.23.0` + vllm-ascend `0.23.0.post1` +
  CANN Toolkit `9.1.0` + torch `2.10.0` + torch-npu `2.10.0.post4` + Python `3.12`。
  **检出本仓的 `v0.23.0` 标签，不使用 `main`**：`main` 是开发版（0.32.x），
  与插件当前发布版接口不匹配；仓内已有 `v0.23.0` 标签，检出后既满足"在本仓"，
  又与插件配套。
- **权重**：`allenai/OLMo-1B-hf`，仓库内是 fp32（4.4 GB），NPU 上需转 bf16。

## 实现路径

1. **环境对齐**：在 CANNLab 中确认镜像自带的 CANN / 驱动版本，选出可用的
   vLLM + vllm-ascend 版本对，记录到文档。
2. **加载验证**
   - 先尝试 `--model-impl transformers`，让 OLMo 走 Transformers 建模后端；
   - 若 NPU 的 attention / 算子路径可用，直接进入第 4 步；
   - 若不可用，在 vLLM 侧补 `olmo.py` 与 registry 注册后重试。
3. **权重加载**：`--dtype bfloat16`，确认权重完整落到 NPU（`npu-smi info` 看 HBM）。
4. **服务化**：`vllm serve <模型路径> --dtype bfloat16 --max-model-len 2048`。
5. **验证**：`POST /v1/completions`（base 模型无 chat template），记录输出、HBM 占用、吞吐。

## 风险与应对

| 风险 | 应对 |
| --- | --- |
| OLMo 不在 vllm-ascend 支持列表，Transformers 后端在 NPU 上可能跑不通 | 第 2 步先验证加载路径，必要时补 `olmo.py` 与 registry 注册 |
| 本仓 main 与 vllm-ascend 发布版本不匹配 | 检出 `v0.23.0` 标签，与插件 0.23.0 整组配套 |
| CANNLab 100 小时用尽 | 环境步骤脚本化，镜像与版本固定，便于重建 |
| 云端环境无持久化 | 代码与文档走 Git，权重每次重下 |
| 无法本地复现 | 所有验证记录带环境版本号与 `npu-smi` 输出 |

## 交付物

- `docs/models/olmo-1b-ascend.md`：版本矩阵、启动命令、实测数据、常见错误
- 至少一次成功推理的运行记录（日志 / 截图）
- 实现 PR 关联 Issue #11

## 验收对照

- [ ] 在昇腾环境加载 `allenai/OLMo-1B-hf` 并完成至少一次推理
- [ ] 有可复现的启动命令与简短使用说明
- [ ] 实现 PR 关联本 Issue 并通过维护者审核合入
