#pragma once

// Copyright 2026 apxinf contributors.
// W4A16 group-32 asymmetric quantized GEMM (compressed-tensors pack-quantized).
//
// Weight layout (all on device):
//   W: int32 [N, K/8]     — 8 int4 per int32, low nibble first (signed via -8)
//   S: bf16  [N, K/32]    — per (output, group) scale
//   Z: int32 [N/8, K/32]  — 8 int8 per int32, packed along the output dim
// dequant: w[n,k] = (nibble(W[n,k/8], k&7) - 8 - zp(n, k/32)) * scale(n, k/32)
// output: C[m,n] = sum_k A[m,k] * w[n,k], A/C are bf16.

#define W4A16_BM 64
#define W4A16_BN 64
#define W4A16_GROUP 32


__global__ void w4a16_dequant_bf16_kernel(
    const int32_t* __restrict__ W,
    const __nv_bfloat16* __restrict__ S,
    const int32_t* __restrict__ Z,
    __nv_bfloat16* __restrict__ out,
    int N, int K) {
  const int64_t total = static_cast<int64_t>(N) * K;
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const int n = static_cast<int>(idx % N);
  const int k = static_cast<int>(idx / N);
  const int groups = K / W4A16_GROUP;
  const int g = k / W4A16_GROUP;
  const int packed = W[static_cast<size_t>(n) * (K / 8) + (k >> 3)];
  const int nib = (packed >> (4 * (k & 7))) & 0xF;
  const int zp_word = Z[static_cast<size_t>(n / 8) * groups + g];
  const int zp = ((zp_word >> (4 * (n & 7))) & 0xF) - 8;
  const float scale = __bfloat162float(S[static_cast<size_t>(n) * groups + g]);
  out[static_cast<size_t>(k) * N + n] = __float2bfloat16((static_cast<float>(nib - 8 - zp)) * scale);
}


// Decode-specialized W4A16 GEMM.  Decode has only a handful of rows, so the
// general 64x64 kernel wastes most of its threads on padded activation rows.
// One thread owns one output column and accumulates up to eight rows while a
// block cooperatively stages the current group of activation values.
__global__ void w4a16_gemm_decode_kernel(
    const __nv_bfloat16* __restrict__ A,
    const int32_t* __restrict__ W,
    const __nv_bfloat16* __restrict__ S,
    const int32_t* __restrict__ Z,
    __nv_bfloat16* __restrict__ C,
    int M, int N, int K) {
  constexpr int TILE_N = 128;
  __shared__ float As[8][W4A16_GROUP];
  const int tid = threadIdx.x;
  const int n = blockIdx.x * TILE_N + tid;
  const int groups = K / W4A16_GROUP;
  const int w_row_stride = K / 8;
  float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f,
                  0.0f, 0.0f, 0.0f, 0.0f};

  for (int g = 0; g < groups; ++g) {
    const int k0 = g * W4A16_GROUP;
    for (int i = tid; i < M * W4A16_GROUP; i += TILE_N) {
      const int m = i / W4A16_GROUP;
      const int c = i % W4A16_GROUP;
      As[m][c] = __bfloat162float(A[static_cast<size_t>(m) * K + k0 + c]);
    }
    __syncthreads();

    float scale = 0.0f;
    int zp = 0;
    if (n < N) {
      const int zp_word = Z[static_cast<size_t>(n / 8) * groups + g];
      zp = ((zp_word >> (4 * (n & 7))) & 0xF) - 8;
      scale = __bfloat162float(S[static_cast<size_t>(n) * groups + g]);
      for (int c = 0; c < W4A16_GROUP; ++c) {
        const int k = k0 + c;
        const int packed = W[static_cast<size_t>(n) * w_row_stride + (k >> 3)];
        const int nib = (packed >> (4 * (k & 7))) & 0xF;
        const float w = static_cast<float>(nib - 8 - zp) * scale;
        for (int m = 0; m < M; ++m) {
          acc[m] += As[m][c] * w;
        }
      }
    }
    __syncthreads();
  }

  if (n < N) {
    for (int m = 0; m < M; ++m) {
      C[static_cast<size_t>(m) * N + n] = __float2bfloat16(acc[m]);
    }
  }
}

__global__ void w4a16_gemm_kernel(
    const __nv_bfloat16* __restrict__ A,  // [M, K]
    const int32_t* __restrict__ W,        // [N, K/8]
    const __nv_bfloat16* __restrict__ S,  // [N, K/32]
    const int32_t* __restrict__ Z,        // [N/8, K/32]
    __nv_bfloat16* __restrict__ C,        // [M, N]
    int M, int N, int K) {
  __shared__ float As[W4A16_BM][W4A16_GROUP];
  __shared__ float Ws[W4A16_BN][W4A16_GROUP];
  __shared__ float zs[W4A16_BN];
  __shared__ float ss[W4A16_BN];

  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;  // 256 (16x16)
  const int m0 = blockIdx.x * W4A16_BM;
  const int n0 = blockIdx.y * W4A16_BN;

  const int ngroups = K / W4A16_GROUP;
  const int w_row_stride = K / 8;

  // Each thread owns a 4x4 sub-tile: row = tid/16*4 + i, col = tid%16*4 + j.
  const int tr = (tid / 16) * 4;
  const int tc = (tid % 16) * 4;
  float acc[4][4] = {0.0f};

  for (int g = 0; g < ngroups; ++g) {
    const int k0 = g * W4A16_GROUP;

    // Load activation tile [BM, 32] bf16 -> f32.
    for (int i = tid; i < W4A16_BM * W4A16_GROUP; i += nthreads) {
      const int r = i / W4A16_GROUP;
      const int c = i % W4A16_GROUP;
      const int m = m0 + r;
      const int k = k0 + c;
      As[r][c] = (m < M && k < K) ? __bfloat162float(A[(size_t)m * K + k]) : 0.0f;
    }

    // Load scale + zero-point for the BN output columns.
    for (int i = tid; i < W4A16_BN; i += nthreads) {
      const int n = n0 + i;
      if (n < N) {
        const int zp = Z[(size_t)(n / 8) * ngroups + g];
        zs[i] = (float)(((zp >> (4 * (n & 7))) & 0xF) - 8);
        ss[i] = __bfloat162float(S[(size_t)n * ngroups + g]);
      } else {
        zs[i] = 0.0f;
        ss[i] = 0.0f;
      }
    }

    __syncthreads();  // publish scales/zero-points before dequant readers
    // Dequant weight tile [BN, 32].
    for (int i = tid; i < W4A16_BN * W4A16_GROUP; i += nthreads) {
      const int r = i / W4A16_GROUP;
      const int c = i % W4A16_GROUP;
      const int n = n0 + r;
      const int k = k0 + c;
      if (n < N && k < K) {
        const int wp = W[(size_t)n * w_row_stride + (k >> 3)];
        const int nib = (wp >> (4 * (k & 7))) & 0xF;
        Ws[r][c] = ((float)(nib - 8) - zs[r]) * ss[r];
      } else {
        Ws[r][c] = 0.0f;
      }
    }
    __syncthreads();

    // Accumulate: 32-element inner loop over shared tiles.
    #pragma unroll
    for (int kk = 0; kk < W4A16_GROUP; ++kk) {
      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float a = As[tr + i][kk];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
          acc[i][j] += a * Ws[tc + j][kk];
        }
      }
    }
    __syncthreads();
  }

  // Write out.
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int m = m0 + tr + i;
    if (m >= M) continue;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int n = n0 + tc + j;
      if (n < N) C[(size_t)m * N + n] = __float2bfloat16(acc[i][j]);
    }
  }
}
