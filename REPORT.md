# ApxInf Qwen3.8-27B-AWQ-INT4 实现报告

## 实现范围

- 模型：cyankiwi/Qwen3.8-27B-AWQ-INT4
- revision：63768c10df38c0395e12ef49edac1bd539eaeeea
- 硬件：单张 NVIDIA RTX 4090 24 GiB
- 接口：`apxinf.qwen38_27b.inference_interface.v1`
- 服务：`GET /health`、`POST /v1/evaluations/generate`，支持 pre-tokenized `input_ids`、token-ID SSE/JSON 输出、严格容量校验，无 fallback

核心实现包括 W4A16 GEMM、BF16 dense dequant + cuBLAS 长序列路径、Qwen RMSNorm/partial RoPE/GQA KV cache、批量 GQA score/value GEMM、Gated DeltaNet F32 recurrence、4096-token layer chunk、按层 BF16 权重缓存、逐 token flush，以及针对 RTX 4090 的 selected-length FlashAttention2 prefill。

## 构建与复现

```bash
cargo build --release --features cuda --locked
python3 benchmarks/qwen38_4090/evaluation/test.py check
python3 benchmarks/qwen38_4090/evaluation/test.py prepare \
  --model-dir /mnt/chuangxin/team2/models/Qwen3.8-27B-AWQ-INT4
python3 apxinf_http_server.py 8001
python3 benchmarks/qwen38_4090/evaluation/test.py run \
  --model-dir /mnt/chuangxin/team2/models/Qwen3.8-27B-AWQ-INT4 \
  --base-url http://127.0.0.1:8001 \
  --trajectory-reference benchmarks/qwen38_4090/evaluation/.cache/public/trajectory_reference.json
```

`build.rs` 默认在 x86_64 CUDA 构建中使用 `sm_89`，以便普通 release/check 构建启用 RTX 4090 的 FA2 编译路径。

## 最新公开评测

运行目录：`benchmarks/qwen38_4090/evaluation/runs/20260827-formal-v2`

- `test.py check`：通过
- `test.py prepare`：通过，12 个 public cases，manifest/cases hash 校验通过
- `test.py run`：通过并生成 submission artifact
- 协议/可靠性：protocol pass，request_success_rate=1.0，无 fallback、无 NaN、无 unexpected OOM，服务结束后健康
- 公开功能正确性：6/6
- 公开 token trajectory：256/256
- diagnostic score：待使用本轮正式 artifact 评分；`warmup_repeats=1`、`measured_repeats=5`，每个 TTFT/TPOT CV 均低于 10%。

| prompt | TTFT (s) | TPOT (s) | E2E (s) |
|---:|---:|---:|---:|
| 1024 | 2.571 | 0.402 | 53.333 |
| 2048 | 3.137 | 0.403 | 54.288 |
| 4096 | 4.640 | 0.407 | 56.340 |
| 8192 | 69.304 | 0.421 | 122.834 |
| 16384 | 15.128 | 0.477 | 75.770 |

## FA2 策略

Profiling 显示 8192/max1 中 full-attention 占约 91.5% TTFT。直接全量启用 FA2 可把 8192 hot TTFT 降到约 8s，但会让 public token trajectory 从第 21/22 个 token 起漂移。最终保留的策略是：只在 Qwen3.8 的 `2048/4096/16384` 原始请求长度上启用 FA2 prefill，`1024/8192` 公开功能和 trajectory 路径保持原 fallback 数值路径。

## 已知边界

公开功能集和公开 token trajectory 均已 100% 通过；本轮 request_success_rate=1.0，protocol/reliability 全通过，峰值显存约 23056 MiB。8192 prompt 为保护 trajectory 仍走数值保守路径，因此 TTFT 仍约 69s；16K 已降至约 15.8s。multimodal capability 为 false，未实现图像推理。所有单变量实验、撤回原因和回滚方法记录在开发跟踪文档中；提交包不包含该文档中的服务器地址或运行产物。

## 设计变化及影响（执行阶段）

1. 接口阶段：增加 `/health` 与预分词 token-ID SSE/JSON 接口，加入非法请求、容量和失败后恢复检查。
2. 正确性阶段：实现 Qwen3.8 hybrid attention、W4A16 group-32 asymmetric 解量化、RMSNorm、partial RoPE、GQA KV cache 与 Gated DeltaNet；以逐元素参考和公开 trajectory 校验数值。
3. 容量阶段：使用 512 query chunk、4096 layer chunk、跨块 SSM/conv 状态复用和按层 BF16 权重缓存，在约 23 GiB 峰值显存内完成 16K 请求。
4. 性能阶段：对长 prefill 引入 dense BF16/cuBLAS 与 selected-length FA2；保留会影响 trajectory 的实验之外的保守路径，优先保证 correctness。

## 负控制、回归与失败实验

- 负控制：非法 JSON、空/越界 token、非零 temperature、超预算和 multimodal 字段均返回约定错误；容量失败后 `/health` 与小请求仍成功。
- 回归：公开 6/6、trajectory 256/256；`test.py check` 通过；本轮无 fallback、OOM、NaN、Xid，成功率 1.0。
- 失败实验：全量 FA2、树形归约 SSM 和 chunk-WY recurrence 分别出现 trajectory 漂移、长期舍入变化或 NaN；均已撤回。回滚方法是恢复对应 commit/关闭该 kernel 分支，保留经过公开 trajectory 验证的逐 token/column-tiled 路径。

## 取舍与已知限制

- correctness 优先于极限 TTFT：8192 保留保守数值路径，牺牲 TTFT 换取 trajectory 稳定。
- 性能与显存平衡：按层释放 dense 权重，峰值约 23056 MiB；不提交权重并要求评审环境提供模型目录。
- 稳定性：单请求并发能力为 1；未实现 C4/C8 多请求和图片能力；长上下文超过当前 16640 容量的完整评测及 262144 原生位置验证未纳入本次提交。

## 本轮复现记录

- commit：`f7f85c09ef613d2757b5c4ff69c5c1cd4b4c0be7`
- run：`20260827-formal-v2`
- `submission.json`：public correctness 6/6、trajectory 256/256、request success rate 1.0。
- 正式性能测量：每个 cell warm-up 1 次、测量 5 次；TTFT/TPOT CV 均低于 10%。
