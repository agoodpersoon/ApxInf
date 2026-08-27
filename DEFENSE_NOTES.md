# 现场答辩提纲

## 1. 问题与目标

在单张 RTX 4090 上完成 Qwen3.8-27B AWQ INT4 的协议兼容、公开正确性和稳定推理；重点是 hybrid attention、Gated DeltaNet 与长 prompt 显存约束。

## 2. 核心设计

- W4A16 group-32 asymmetric 解量化；短请求使用融合 kernel，长 prefill 使用按层 dense BF16/cuBLAS。
- Full attention 使用 GQA KV cache、partial RoPE 和 selected-length FA2；linear attention 使用 F32 Gated DeltaNet recurrence。
- 通过 query/layer chunk、状态跨块复用、按层释放权重和逐 token flush 控制显存及客户端 TTFT/TPOT。

## 3. 评测演示

展示 `/health` 的 contract、revision、fallback 和 capability；运行 `test.py check`；展示最新 `submission.json` 的 6/6、256/256、成功率 1.0；再展示 1K/2K/4K/8K/16K TTFT/TPOT 表。

## 4. 工程取舍

全量 FA2 和激进 chunk-WY 虽可降低局部延迟，但会造成 trajectory 漂移或 NaN，因此最终保留 correctness-first 路径。峰值显存约 23 GiB，multimodal、多请求和超过 16640 的长上下文属于已知限制。

## 5. 合规说明

输出由模型和通用 kernel 路径产生，没有针对 case ID、公开 token 序列或答案硬编码；提交包不含权重、凭据、机器地址和未公开数据。
