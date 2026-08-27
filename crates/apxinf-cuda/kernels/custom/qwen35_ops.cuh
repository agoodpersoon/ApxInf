#pragma once

// Copyright 2026 apxinf contributors.
// Small Qwen3.8-specific elementwise/rope helpers.

// ── sigmoid ────────────────────────────────────────────────────────────────
__global__ void sigmoid_bf16_kernel(
    const __nv_bfloat16* input, __nv_bfloat16* output, int64_t count) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) return;
  float value = __bfloat162float(input[index]);
  output[index] = __float2bfloat16(1.0f / (1.0f + expf(-value)));
}

// ── partial RoPE (rotates first rotary_dim of head_dim, rest pass-through) ─
// Input [seq, n_heads, head_dim]; half-split pairs within the rotary region.
__global__ void rope_partial_bf16_kernel(
    const __nv_bfloat16* input, __nv_bfloat16* output,
    uint32_t head_dim, uint32_t n_heads, uint32_t seq_len,
    float rope_theta, uint32_t pos_offset, uint32_t rotary_dim) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t total = seq_len * n_heads * head_dim;
  if (idx >= total) return;

  uint32_t in_head = idx % head_dim;
  uint32_t base = idx - in_head;

  if (in_head >= rotary_dim) {
    output[idx] = input[idx];
    return;
  }

  uint32_t half = rotary_dim / 2;
  uint32_t pair = in_head % half;
  uint32_t seq = base / (n_heads * head_dim);
  uint32_t pos = seq + pos_offset;
  float freq = 1.0f / powf(rope_theta, 2.0f * (float)pair / (float)rotary_dim);
  float angle = (float)pos * freq;
  float cos_val = cosf(angle);
  float sin_val = sinf(angle);

  float x0 = __bfloat162float(input[base + pair]);
  float x1 = __bfloat162float(input[base + half + pair]);
  if (in_head < half) {
    output[idx] = __float2bfloat16(x0 * cos_val - x1 * sin_val);
  } else {
    output[idx] = __float2bfloat16(x0 * sin_val + x1 * cos_val);
  }
}

// ── column-range slice: output[r,c] = input[r, col_off + c] ────────────────
// HF reshapes q_proj output as [heads, 2*head_dim] and chunks the
// last dimension, so query and gate are interleaved per head.
__global__ void split_q_gate_bf16_kernel(
    const __nv_bfloat16* input,
    __nv_bfloat16* query,
    __nv_bfloat16* gate,
    int rows, int n_heads, int head_dim) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = static_cast<int64_t>(rows) * n_heads * head_dim;
  if (idx >= total) return;
  int t = static_cast<int>(idx / (n_heads * head_dim));
  int rem = static_cast<int>(idx % (n_heads * head_dim));
  int h = rem / head_dim;
  int d = rem % head_dim;
  int src = t * (2 * n_heads * head_dim) + h * (2 * head_dim) + d;
  query[idx] = input[src];
  gate[idx] = input[src + head_dim];
}

__global__ void slice_cols_bf16_kernel(
    const __nv_bfloat16* input, __nv_bfloat16* output,
    int rows, int src_cols, int dst_cols, int col_off) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = static_cast<int64_t>(rows) * dst_cols;
  if (index >= total) return;
  int r = static_cast<int>(index / dst_cols);
  int c = static_cast<int>(index % dst_cols);
  output[index] = input[static_cast<int64_t>(r) * src_cols + col_off + c];
}
