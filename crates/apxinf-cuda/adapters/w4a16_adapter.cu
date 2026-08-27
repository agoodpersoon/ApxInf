// Copyright 2026 apxinf contributors.
// Stable C ABI and CUDA launch policy for the W4A16 quantized GEMM.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {
#include "../kernels/custom/w4a16.cuh"
}  // namespace


extern "C" cudaError_t apxinf_static_w4a16_dequant_bf16(
    const void* packed, const void* scale, const void* zero_point,
    void* output, int n, int k, cudaStream_t stream) {
  if (packed == nullptr || scale == nullptr || zero_point == nullptr ||
      output == nullptr || n <= 0 || k <= 0 || (k % 32) != 0 || (n % 8) != 0) {
    return cudaErrorInvalidValue;
  }
  const int64_t count = static_cast<int64_t>(n) * k;
  const int threads = 256;
  const int blocks = static_cast<int>((count + threads - 1) / threads);
  w4a16_dequant_bf16_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<const int32_t*>(packed),
      static_cast<const __nv_bfloat16*>(scale),
      static_cast<const int32_t*>(zero_point),
      static_cast<__nv_bfloat16*>(output), n, k);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_w4a16_gemm_bf16(
    const void* activation, const void* packed, const void* scale,
    const void* zero_point, void* output,
    int m, int n, int k, cudaStream_t stream) {
  if (activation == nullptr || packed == nullptr || scale == nullptr ||
      zero_point == nullptr || output == nullptr ||
      m <= 0 || n <= 0 || k <= 0 || (k % 32) != 0) {
    return cudaErrorInvalidValue;
  }
  if (m <= 8) {
    const int threads = 128;
    const int blocks = (n + 127) / 128;
    w4a16_gemm_decode_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(activation),
        static_cast<const int32_t*>(packed),
        static_cast<const __nv_bfloat16*>(scale),
        static_cast<const int32_t*>(zero_point),
        static_cast<__nv_bfloat16*>(output), m, n, k);
  } else {
    const dim3 grid((m + W4A16_BM - 1) / W4A16_BM, (n + W4A16_BN - 1) / W4A16_BN);
    const int threads = 256;
    w4a16_gemm_kernel<<<grid, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(activation),
        static_cast<const int32_t*>(packed),
        static_cast<const __nv_bfloat16*>(scale),
        static_cast<const int32_t*>(zero_point),
        static_cast<__nv_bfloat16*>(output), m, n, k);
  }
  return cudaGetLastError();
}
