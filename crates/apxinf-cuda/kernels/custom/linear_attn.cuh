#pragma once

// Copyright 2026 apxinf contributors.
// Pure CUDA operators for the Qwen3-Next "Gated DeltaNet" linear attention
// layer (48 layers of Qwen3.8-27B). Launch policy lives under adapters/.
//
// Reference math (HuggingFace transformers torch implementation):
//   1. qkv = in_proj_qkv(x), z = in_proj_z(x), a = in_proj_a(x), b = in_proj_b(x)
//   2. query/key/value = split(qkv, [key_dim, key_dim, value_dim])
//   3. causal depthwise conv1d (kernel=4, left pad=3) on [query|key|value],
//      then silu
//   4. reshape to [seq, heads, head_dim]
//   5. beta = sigmoid(b)
//   6. g = -exp(A_log) * softplus(a + dt_bias)
//   7. repeat_interleave query/key from 16 -> 48 heads (each head ×3)
//   8. L2-normalize query/key along head_dim; query × 1/sqrt(head_dim)
//   9. SSM recurrence in f32 (persistent S [48, 128, 128]):
//        S = S * exp(g_t)
//        kv = Sᵀ @ k_t
//        delta = (v_t - kv) * beta_t
//        S = S + k_t ⊗ delta
//        out = Sᵀ @ q_t
//  10. out = rms_norm(out, head_dim) * silu(z)
//  11. out = out_proj(out)
//
// Correctness is the priority; the kernels below favour a straightforward,
// obviously-correct implementation. Performance notes for future work are
// marked with "TODO(perf)".

// ── Device helpers ────────────────────────────────────────────────────────

__device__ __forceinline__ float delta_net_silu(float x) {
  // silu(x) = x * sigmoid(x)
  return x / (1.0f + expf(-x));
}

__device__ __forceinline__ float delta_net_sigmoid(float x) {
  return 1.0f / (1.0f + expf(-x));
}

// Numerically stable softplus: log(1 + exp(x)).
__device__ __forceinline__ float delta_net_softplus(float x) {
  return fmaxf(x, 0.0f) + log1pf(expf(-fabsf(x)));
}

// ── Causal depthwise conv1d + silu ────────────────────────────────────────
//
// x       : [seq, channels] bf16 (the [query|key|value] concatenation)
// weight  : [channels, kernel] bf16 (contiguous conv1d.weight [channels,1,kernel])
// conv_state: [channels, kernel-1] bf16 in/out — the last kernel-1 conv inputs
// y       : [seq, channels] bf16 = silu(causal_conv(x))
//
// Prefill (seq > 1): full causal conv with `kernel-1` zeroes padded on the
// left, then the persistent history is refreshed with the last kernel-1 rows.
// Decode  (seq == 1, conv_state != nullptr): convolves the stored history
// with the single new input, then rolls the history forward. This matches
// HF's `_conv_state` maintenance on both paths.
//
// TODO(perf): one thread per channel re-reads x `kernel` times per token; a
// shared-memory row tiling would cut HBM traffic by ~4×.

__global__ void delta_net_conv1d_kernel(
    const __nv_bfloat16* x, const __nv_bfloat16* weight,
    __nv_bfloat16* conv_state, __nv_bfloat16* y,
    int seq, int channels, int kernel)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    const int hist = kernel - 1;

    if (seq == 1 && conv_state != nullptr) {
        // Decode: [state[0], state[1], state[2], x[0]] · w, then shift.
        float acc = 0.0f;
        for (int i = 0; i < kernel; ++i) {
            const float xv = (i < hist)
                ? __bfloat162float(conv_state[c * hist + i])
                : __bfloat162float(x[c]);
            acc += __bfloat162float(weight[c * kernel + i]) * xv;
        }
        y[c] = __float2bfloat16(delta_net_silu(acc));
        for (int j = 0; j < hist - 1; ++j)
            conv_state[c * hist + j] = conv_state[c * hist + j + 1];
        conv_state[c * hist + hist - 1] = x[c];
        return;
    }

    // Prefill: causal conv with hist zeroes of left padding.
    for (int t = 0; t < seq; ++t) {
        float acc = 0.0f;
        for (int i = 0; i < kernel; ++i) {
            const int src = t - hist + i;
            const float xv = (src >= 0)
                ? __bfloat162float(x[static_cast<int64_t>(src) * channels + c])
                : (conv_state != nullptr
                    ? __bfloat162float(conv_state[c * hist + (src + hist)])
                    : 0.0f);
            acc += __bfloat162float(weight[c * kernel + i]) * xv;
        }
        y[static_cast<int64_t>(t) * channels + c] =
            __float2bfloat16(delta_net_silu(acc));
    }

    // Keep the persistent history in sync for the next decode step.
    if (conv_state != nullptr) {
        for (int j = 0; j < hist; ++j) {
            const int src = seq - hist + j;
            const float v = (src >= 0)
                ? __bfloat162float(x[static_cast<int64_t>(src) * channels + c])
                : 0.0f;
            conv_state[c * hist + j] = __float2bfloat16(v);
        }
    }
}

// ── Split [query|key|value] (unequal widths) ──────────────────────────────
//
// fused : [seq, qd + kd + vd] bf16
// q/k/v : [seq, qd] / [seq, kd] / [seq, vd] bf16
// The standard `qkv_split_bias_bf16_kernel` only handles three equal widths;
// this layer splits 2048 | 2048 | 6144.

__global__ void delta_net_split_qkv_kernel(
    const __nv_bfloat16* fused,
    __nv_bfloat16* q, __nv_bfloat16* k, __nv_bfloat16* v,
    int seq, int qd, int kd, int vd)
{
    const int total = qd + kd + vd;
    const int64_t count = static_cast<int64_t>(seq) * total;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    const int t = static_cast<int>(idx / total);
    const int col = static_cast<int>(idx - static_cast<int64_t>(t) * total);
    const float val = __bfloat162float(fused[static_cast<int64_t>(t) * total + col]);
    if (col < qd) {
        q[t * qd + col] = __float2bfloat16(val);
    } else if (col < qd + kd) {
        k[t * kd + col - qd] = __float2bfloat16(val);
    } else {
        v[t * vd + col - qd - kd] = __float2bfloat16(val);
    }
}

// ── Gate/scale preparation ────────────────────────────────────────────────
//
// q, k : [seq, n_heads, head_dim] bf16
// a, b : [seq, v_heads] bf16
// a_log, dt_bias : [v_heads] bf16
// beta, g : [seq, v_heads] f32 (out) — kept in f32 like the HF reference
// q_r, k_r : [seq, v_heads, head_dim] f32 (out)
//
// One block per (v_head, token); head_dim threads. Value head h mirrors
// source key head h * n_heads / v_heads (repeat_interleave of each of the
// n_heads heads by v_heads/n_heads).

__global__ void delta_net_prepare_kernel(
    const __nv_bfloat16* q, const __nv_bfloat16* k,
    const __nv_bfloat16* a, const __nv_bfloat16* b,
    const __nv_bfloat16* a_log, const __nv_bfloat16* dt_bias,
    float* beta, float* g,
    float* q_r, float* k_r,
    int seq, int n_heads, int v_heads, int head_dim, float scale)
{
    const int h = blockIdx.x;
    const int t = blockIdx.y;
    if (h >= v_heads || t >= seq) return;
    const int d = threadIdx.x;
    // Launch contract: blockDim.x == head_dim, so this guard never fires.
    if (d >= head_dim) return;

    const int sh = h * n_heads / v_heads;
    const int qk_src = (t * n_heads + sh) * head_dim + d;
    const float qv = __bfloat162float(q[qk_src]);
    const float kv = __bfloat162float(k[qk_src]);

    // L2 norms of the query and key heads over head_dim (warp shuffle + a
    // cross-warp shared reduction).
    float qsq = qv * qv;
    float ksq = kv * kv;
    for (int off = 16; off > 0; off >>= 1) {
        qsq += __shfl_xor_sync(0xffffffff, qsq, off);
        ksq += __shfl_xor_sync(0xffffffff, ksq, off);
    }
    __shared__ float warp_q[8];
    __shared__ float warp_k[8];
    const int warp = d / 32;
    const int lane = d % 32;
    if (lane == 0) {
        warp_q[warp] = qsq;
        warp_k[warp] = ksq;
    }
    __syncthreads();
    if (warp == 0) {
        const int nwarps = head_dim / 32;
        float qs = (d < nwarps) ? warp_q[d] : 0.0f;
        float ks = (d < nwarps) ? warp_k[d] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) {
            qs += __shfl_xor_sync(0xffffffff, qs, off);
            ks += __shfl_xor_sync(0xffffffff, ks, off);
        }
        if (lane == 0) {
            warp_q[0] = qs;
            warp_k[0] = ks;
        }
    }
    __syncthreads();

    // FLA-aligned L2 norm (matches HF l2norm + rsqrt(sum_sq + 1e-6)).
    const float q_inv = rsqrtf(warp_q[0] + 1e-6f);
    const float k_inv = rsqrtf(warp_k[0] + 1e-6f);

    const int dst = (t * v_heads + h) * head_dim + d;
    q_r[dst] = qv * q_inv * scale;
    k_r[dst] = kv * k_inv;

    if (d == 0) {
        const float a_val = __bfloat162float(a[t * v_heads + h]);
        const float b_val = __bfloat162float(b[t * v_heads + h]);
        beta[t * v_heads + h] = delta_net_sigmoid(b_val);
        const float log_decay = __bfloat162float(a_log[h]);
        const float dt = __bfloat162float(dt_bias[h]);
        g[t * v_heads + h] = -expf(log_decay) * delta_net_softplus(a_val + dt);
    }
}

// ── SSM recurrence step ───────────────────────────────────────────────────
//
// q_r, k_r : f32; v : bf16
// beta, g     : [seq, v_heads] f32 (per-head scalars per token)
// ssm_state   : [v_heads, head_dim, head_dim] f32, in/out persistent state
// out         : [seq, v_heads, head_dim] bf16
//
// One block per value head; HEAD_DIM threads. Thread `tid` owns the column
// S[h][·][tid] of the persistent state and keeps it in HEAD_DIM registers,
// so the whole recurrence for a token is register-local:
//   kv[tid]   = Σ_k S[k][tid] · k_t[k]
//   delta     = (v_t[tid] - kv[tid]) · beta_t
//   S[k][tid] += k_t[k] · delta
//   out[tid]  = Σ_k S[k][tid] · q_t[k]
// k_t and q_t are staged in shared memory once per token; the token loop is
// naturally serial because every update depends on the previous token's S.
//
// TODO(perf): for long prefills this can be replaced by a chunked parallel
// scan (associative combination of the (decay, rank-1 update) linear map),
// which is what production Gated DeltaNet kernels use. The current serial
// per-head loop is the reference-correct implementation.

template<int HEAD_DIM>
__global__ void delta_net_step_kernel(
    const float* q_r, const float* k_r,
    const __nv_bfloat16* v, const float* beta, const float* g,
    float* ssm_state, __nv_bfloat16* out,
    int seq, int v_heads)
{
    const int h = blockIdx.x;
    if (h >= v_heads) return;
    const int tid = threadIdx.x;
    // Launch contract: blockDim.x == HEAD_DIM.
    if (tid >= HEAD_DIM) return;

    // Load my column of S[h] into registers.
    float s[HEAD_DIM];
    float* S = ssm_state + h * HEAD_DIM * HEAD_DIM;
    #pragma unroll
    for (int k = 0; k < HEAD_DIM; ++k) s[k] = S[k * HEAD_DIM + tid];

    __shared__ float k_sh[HEAD_DIM];
    __shared__ float q_sh[HEAD_DIM];

    for (int t = 0; t < seq; ++t) {
        const int base = (t * v_heads + h) * HEAD_DIM;

        // Stage this token's K/Q for the whole block. Two barriers per token
        // give a clean epoch structure: readers of iteration t finish before
        // writers of iteration t+1 overwrite the buffers (WAR safety).
        __syncthreads();
        k_sh[tid] = k_r[base + tid];
        q_sh[tid] = q_r[base + tid];
        __syncthreads();

        // S = S * exp(g_t)          (g_t <= 0 always, so decay ∈ (0, 1])
        const float decay = expf(g[t * v_heads + h]);
        #pragma unroll
        for (int k = 0; k < HEAD_DIM; ++k) s[k] *= decay;

        // kv[v] = Σ_k S[k][v] · k_t[k]
        float kv = 0.0f;
        #pragma unroll
        for (int k = 0; k < HEAD_DIM; ++k) kv += s[k] * k_sh[k];

        // delta[v] = (v_t[v] - kv[v]) * beta_t
        const float beta_t = beta[t * v_heads + h];
        const float delta = (__bfloat162float(v[base + tid]) - kv) * beta_t;

        // S = S + k_t[k] * delta[v]  (rank-1 update)
        #pragma unroll
        for (int k = 0; k < HEAD_DIM; ++k) s[k] += k_sh[k] * delta;

        // out[v] = Σ_k S[k][v] · q_t[k]
        float acc = 0.0f;
        #pragma unroll
        for (int k = 0; k < HEAD_DIM; ++k) acc += s[k] * q_sh[k];

        out[base + tid] = __float2bfloat16(acc);
    }

    // Persist the updated column.
    #pragma unroll
    for (int k = 0; k < HEAD_DIM; ++k) *(S + k * HEAD_DIM + tid) = s[k];
}


// Column-tiled recurrent update for long prefills.
// Each 32-thread block owns 32 value columns of one head. Unlike the
// reduction-based tile kernel below, every thread owns a complete state
// column, so the two dot products remain register-local. This raises the
// number of independent blocks from 48 to 192 without changing the f32
// recurrence or its update order.
template<int HEAD_DIM, int VALUE_TILE>
__global__ void delta_net_step_columns_kernel(
    const float* q_r, const float* k_r,
    const __nv_bfloat16* v, const float* beta, const float* g,
    float* ssm_state, __nv_bfloat16* out,
    int seq, int v_heads)
{
    constexpr int TILES = HEAD_DIM / VALUE_TILE;
    const int block = blockIdx.x;
    const int h = block / TILES;
    const int tile = block - h * TILES;
    if (h >= v_heads) return;

    const int tid = threadIdx.x;
    if (tid >= VALUE_TILE) return;
    const int vcol = tile * VALUE_TILE + tid;
    float s[HEAD_DIM];
    float* S = ssm_state + h * HEAD_DIM * HEAD_DIM;
    #pragma unroll
    for (int k = 0; k < HEAD_DIM; ++k)
        s[k] = S[k * HEAD_DIM + vcol];

    __shared__ float k_sh[HEAD_DIM];
    __shared__ float q_sh[HEAD_DIM];

    for (int t = 0; t < seq; ++t) {
        const int base = (t * v_heads + h) * HEAD_DIM;
        for (int d = tid; d < HEAD_DIM; d += VALUE_TILE) {
            k_sh[d] = k_r[base + d];
            q_sh[d] = q_r[base + d];
        }
        __syncthreads();

        const float decay = expf(g[t * v_heads + h]);
        #pragma unroll
        for (int k = 0; k < HEAD_DIM; ++k) s[k] *= decay;

        float kv = 0.0f;
        #pragma unroll
        for (int k = 0; k < HEAD_DIM; ++k) kv += s[k] * k_sh[k];
        const float beta_t = beta[t * v_heads + h];
        const float delta = (__bfloat162float(v[base + vcol]) - kv) * beta_t;

        #pragma unroll
        for (int k = 0; k < HEAD_DIM; ++k) s[k] += k_sh[k] * delta;

        float acc = 0.0f;
        #pragma unroll
        for (int k = 0; k < HEAD_DIM; ++k) acc += s[k] * q_sh[k];
        out[base + vcol] = __float2bfloat16(acc);
        __syncthreads();
    }

    #pragma unroll
    for (int k = 0; k < HEAD_DIM; ++k)
        S[k * HEAD_DIM + vcol] = s[k];
}


// Tiled recurrent update for long prefills.
//
// The recurrence is serial in time, but independent across the V columns of
// each head. The reference kernel assigns one whole V column to a thread,
// leaving only 48 resident blocks. This variant assigns eight V columns to a
// 128-thread block (one thread per K row), so all 16 V tiles per head can run
// concurrently. It uses a tree reduction for the two K dot products; decode
// keeps the reference kernel to preserve its bit-level trajectory.
template<int HEAD_DIM, int VALUE_TILE>
__global__ void delta_net_step_tiled_kernel(
    const float* q_r, const float* k_r,
    const __nv_bfloat16* v, const float* beta, const float* g,
    float* ssm_state, __nv_bfloat16* out,
    int seq, int v_heads)
{
    constexpr int TILES = HEAD_DIM / VALUE_TILE;
    const int block = blockIdx.x;
    const int h = block / TILES;
    const int tile = block - h * TILES;
    if (h >= v_heads) return;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int v0 = tile * VALUE_TILE;
    float state[VALUE_TILE];
    float* S = ssm_state + h * HEAD_DIM * HEAD_DIM;
    #pragma unroll
    for (int j = 0; j < VALUE_TILE; ++j)
        state[j] = S[tid * HEAD_DIM + v0 + j];

    __shared__ float k_sh[HEAD_DIM];
    __shared__ float q_sh[HEAD_DIM];
    __shared__ float warp_dot[4][VALUE_TILE];
    __shared__ float dot[VALUE_TILE];

    for (int t = 0; t < seq; ++t) {
        const int base = (t * v_heads + h) * HEAD_DIM;
        k_sh[tid] = k_r[base + tid];
        q_sh[tid] = q_r[base + tid];
        __syncthreads();

        const float decay = expf(g[t * v_heads + h]);
        #pragma unroll
        for (int j = 0; j < VALUE_TILE; ++j)
            state[j] *= decay;

        float kv[VALUE_TILE];
        #pragma unroll
        for (int j = 0; j < VALUE_TILE; ++j)
            kv[j] = state[j] * k_sh[tid];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            #pragma unroll
            for (int j = 0; j < VALUE_TILE; ++j)
                kv[j] += __shfl_down_sync(0xffffffff, kv[j], off);
        }
        if (lane == 0) {
            #pragma unroll
            for (int j = 0; j < VALUE_TILE; ++j)
                warp_dot[warp][j] = kv[j];
        }
        __syncthreads();

        if (tid < 32) {
            #pragma unroll
            for (int j = 0; j < VALUE_TILE; ++j) {
                float value = tid < 4 ? warp_dot[tid][j] : 0.0f;
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1)
                    value += __shfl_down_sync(0xffffffff, value, off);
                if (tid == 0) dot[j] = value;
            }
        }
        __syncthreads();

        const float beta_t = beta[t * v_heads + h];
        #pragma unroll
        for (int j = 0; j < VALUE_TILE; ++j) {
            const float delta =
                (__bfloat162float(v[base + v0 + j]) - dot[j]) * beta_t;
            state[j] += k_sh[tid] * delta;
        }

        float acc[VALUE_TILE];
        #pragma unroll
        for (int j = 0; j < VALUE_TILE; ++j)
            acc[j] = state[j] * q_sh[tid];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            #pragma unroll
            for (int j = 0; j < VALUE_TILE; ++j)
                acc[j] += __shfl_down_sync(0xffffffff, acc[j], off);
        }
        if (lane == 0) {
            #pragma unroll
            for (int j = 0; j < VALUE_TILE; ++j)
                warp_dot[warp][j] = acc[j];
        }
        __syncthreads();
        if (tid < 32) {
            #pragma unroll
            for (int j = 0; j < VALUE_TILE; ++j) {
                float value = tid < 4 ? warp_dot[tid][j] : 0.0f;
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1)
                    value += __shfl_down_sync(0xffffffff, value, off);
                if (tid == 0)
                    out[base + v0 + j] = __float2bfloat16(value);
            }
        }
        // Ensure every thread has finished reading shared q/k and warp
        // reductions before the next token overwrites them.
        __syncthreads();
    }

    #pragma unroll
    for (int j = 0; j < VALUE_TILE; ++j)
        S[tid * HEAD_DIM + v0 + j] = state[j];
}

// ── Output gate: RMSNorm(head_dim) × silu(z) ──────────────────────────────
//
// out, z : [seq, v_heads, head_dim] bf16
// norm_w : [head_dim] bf16
// output : [seq, v_heads, head_dim] bf16
//
// One block per (v_head, token); head_dim threads. Each of the v_heads
// groups is RMS-normalized independently with the shared norm weight.

__global__ void delta_net_norm_gate_kernel(
    const __nv_bfloat16* out, const __nv_bfloat16* z,
    const __nv_bfloat16* norm_w, __nv_bfloat16* output,
    int seq, int v_heads, int head_dim, float eps)
{
    const int h = blockIdx.x;
    const int t = blockIdx.y;
    if (h >= v_heads || t >= seq) return;
    const int d = threadIdx.x;
    // Launch contract: blockDim.x == head_dim.
    if (d >= head_dim) return;

    const int base = (t * v_heads + h) * head_dim;
    const float x = __bfloat162float(out[base + d]);

    // RMS over head_dim (warp shuffle + cross-warp shared reduction).
    float sq = x * x;
    for (int off = 16; off > 0; off >>= 1) sq += __shfl_xor_sync(0xffffffff, sq, off);
    __shared__ float warp_sums[8];
    const int warp = d / 32;
    const int lane = d % 32;
    if (lane == 0) warp_sums[warp] = sq;
    __syncthreads();
    if (warp == 0) {
        const int nwarps = head_dim / 32;
        float s = (d < nwarps) ? warp_sums[d] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffff, s, off);
        if (lane == 0) warp_sums[0] = s;
    }
    __syncthreads();

    const float rms = rsqrtf(warp_sums[0] / (float)head_dim + eps);
    const float w = __bfloat162float(norm_w[d]);
    const float zg = __bfloat162float(z[base + d]);
    output[base + d] = __float2bfloat16(x * rms * w * delta_net_silu(zg));
}
