// Copyright 2026 apxinf contributors.
// Stable C ABI and CUDA launch policy for the Qwen3-Next "Gated DeltaNet"
// linear attention operators (kernels/custom/linear_attn.cuh).
//
// All persistent state (SSM state, conv history) is caller-owned and passed
// as raw pointers; these wrappers never allocate device memory.

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <cmath>
#include <cstdint>

#include "../kernels/custom/linear_attn.cuh"

#define DELTA_NET_BLOCK 256

extern "C" cudaError_t apxinf_static_delta_net_conv1d(
    const void* x, const void* weight, void* conv_state, void* y,
    int seq, int channels, int kernel, cudaStream_t stream)
{
    if (seq <= 0 || channels <= 0 || kernel <= 1) return cudaErrorInvalidValue;
    dim3 grid((channels + DELTA_NET_BLOCK - 1) / DELTA_NET_BLOCK, 1, 1);
    dim3 block(DELTA_NET_BLOCK, 1, 1);
    delta_net_conv1d_kernel<<<grid, block, 0, stream>>>(
        (const __nv_bfloat16*)x, (const __nv_bfloat16*)weight,
        (__nv_bfloat16*)conv_state, (__nv_bfloat16*)y,
        seq, channels, kernel);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_delta_net_split_qkv(
    const void* fused, void* q, void* k, void* v,
    int seq, int qd, int kd, int vd, cudaStream_t stream)
{
    if (seq <= 0 || qd <= 0 || kd <= 0 || vd <= 0) return cudaErrorInvalidValue;
    const int64_t total = static_cast<int64_t>(seq) * (qd + kd + vd);
    dim3 grid(static_cast<int>((total + DELTA_NET_BLOCK - 1) / DELTA_NET_BLOCK), 1, 1);
    dim3 block(DELTA_NET_BLOCK, 1, 1);
    delta_net_split_qkv_kernel<<<grid, block, 0, stream>>>(
        (const __nv_bfloat16*)fused, (__nv_bfloat16*)q, (__nv_bfloat16*)k,
        (__nv_bfloat16*)v, seq, qd, kd, vd);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_delta_net_prepare(
    const void* q, const void* k, const void* a, const void* b,
    const void* a_log, const void* dt_bias,
    void* beta, void* g, void* q_r, void* k_r,
    int seq, int n_heads, int v_heads, int head_dim, float scale,
    cudaStream_t stream)
{
    if (seq <= 0 || n_heads <= 0 || v_heads <= 0 || v_heads % n_heads != 0 ||
        head_dim <= 0 || head_dim % 32 != 0 || head_dim > 256)
        return cudaErrorInvalidValue;
    // One block per (value head, token); head_dim threads per block.
    dim3 grid(v_heads, seq, 1);
    dim3 block(head_dim, 1, 1);
    delta_net_prepare_kernel<<<grid, block, 0, stream>>>(
        (const __nv_bfloat16*)q, (const __nv_bfloat16*)k,
        (const __nv_bfloat16*)a, (const __nv_bfloat16*)b,
        (const __nv_bfloat16*)a_log, (const __nv_bfloat16*)dt_bias,
        (float*)beta, (float*)g,
        (float*)q_r, (float*)k_r,
        seq, n_heads, v_heads, head_dim, scale);
    return cudaGetLastError();
}


extern "C" cudaError_t apxinf_static_delta_net_step(
    const void* q_r, const void* k_r, const void* v,
    const void* beta, const void* g,
    void* ssm_state, void* out,
    int seq, int v_heads, int head_dim, cudaStream_t stream)
{
    if (seq <= 0 || v_heads <= 0) return cudaErrorInvalidValue;
    // The recurrence kernel keeps one S column per thread in registers and
    // is instantiated for head_dim == 128 (Qwen3.8-27B configuration).
    if (head_dim != 128) return cudaErrorInvalidConfiguration;
    if (seq >= 64) {
        // Long prefills use four independent column blocks per head. Each
        // 32-thread block keeps complete state columns in registers and
        // avoids the reduction overhead of the V-tiled experiment.
        constexpr int VALUE_TILE = 32;
        const int tiles = head_dim / VALUE_TILE;
        delta_net_step_columns_kernel<128, VALUE_TILE>
            <<<v_heads * tiles, VALUE_TILE, 0, stream>>>(
                (const float*)q_r, (const float*)k_r,
                (const __nv_bfloat16*)v, (const float*)beta,
                (const float*)g, (float*)ssm_state,
                (__nv_bfloat16*)out, seq, v_heads);
    } else {
        delta_net_step_kernel<128><<<v_heads, 128, 0, stream>>>(
            (const float*)q_r, (const float*)k_r,
            (const __nv_bfloat16*)v, (const float*)beta,
            (const float*)g, (float*)ssm_state,
            (__nv_bfloat16*)out, seq, v_heads);
    }
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_delta_net_norm_gate(
    const void* out, const void* z, const void* norm_w, void* output,
    int seq, int v_heads, int head_dim, float eps, cudaStream_t stream)
{
    if (seq <= 0 || v_heads <= 0 || head_dim <= 0 || head_dim % 32 != 0 ||
        head_dim > 256)
        return cudaErrorInvalidValue;
    // One block per (value head, token); head_dim threads per block.
    dim3 grid(v_heads, seq, 1);
    dim3 block(head_dim, 1, 1);
    delta_net_norm_gate_kernel<<<grid, block, 0, stream>>>(
        (const __nv_bfloat16*)out, (const __nv_bfloat16*)z,
        (const __nv_bfloat16*)norm_w, (__nv_bfloat16*)output,
        seq, v_heads, head_dim, eps);
    return cudaGetLastError();
}
