// Copyright 2026 apxinf contributors.
// Stable C ABI for small Qwen3.8-specific elementwise/rope helpers.

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <cstdint>

namespace {
#include "../kernels/custom/qwen35_ops.cuh"
}  // namespace

extern "C" cudaError_t apxinf_qwen35_sigmoid_bf16(
    const void* input, void* output, int64_t count, cudaStream_t stream) {
  if (input == nullptr || output == nullptr || count <= 0) {
    return cudaErrorInvalidValue;
  }
  const int threads = 256;
  const int64_t blocks = (count + threads - 1) / threads;
  sigmoid_bf16_kernel<<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<__nv_bfloat16*>(output), count);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_qwen35_rope_partial_bf16(
    const void* input, void* output,
    uint32_t head_dim, uint32_t n_heads, uint32_t seq_len,
    float rope_theta, uint32_t pos_offset, uint32_t rotary_dim,
    cudaStream_t stream) {
  if (input == nullptr || output == nullptr || head_dim == 0) {
    return cudaErrorInvalidValue;
  }
  const uint32_t total = seq_len * n_heads * head_dim;
  const int threads = 256;
  const uint32_t blocks = (total + threads - 1) / threads;
  rope_partial_bf16_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<__nv_bfloat16*>(output),
      head_dim, n_heads, seq_len, rope_theta, pos_offset, rotary_dim);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_qwen35_split_q_gate_bf16(
    const void* input, void* query, void* gate,
    int rows, int n_heads, int head_dim, cudaStream_t stream) {
  if (input == nullptr || query == nullptr || gate == nullptr ||
      rows <= 0 || n_heads <= 0 || head_dim <= 0) {
    return cudaErrorInvalidValue;
  }
  const int64_t total = static_cast<int64_t>(rows) * n_heads * head_dim;
  const int threads = 256;
  const int64_t blocks = (total + threads - 1) / threads;
  split_q_gate_bf16_kernel<<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<__nv_bfloat16*>(query),
      static_cast<__nv_bfloat16*>(gate),
      rows, n_heads, head_dim);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_qwen35_slice_cols_bf16(
    const void* input, void* output,
    int rows, int src_cols, int dst_cols, int col_off, cudaStream_t stream) {
  if (input == nullptr || output == nullptr || rows <= 0 || dst_cols <= 0) {
    return cudaErrorInvalidValue;
  }
  const int64_t total = static_cast<int64_t>(rows) * dst_cols;
  const int threads = 256;
  const int64_t blocks = (total + threads - 1) / threads;
  slice_cols_bf16_kernel<<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<__nv_bfloat16*>(output), rows, src_cols, dst_cols, col_off);
  return cudaGetLastError();
}
