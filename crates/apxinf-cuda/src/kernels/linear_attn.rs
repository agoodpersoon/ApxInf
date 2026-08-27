//! Qwen3-Next "Gated DeltaNet" linear attention operator.
//!
//! One fused forward pass for the 48 linear-attention layers of
//! Qwen3.8-27B. The pipeline mirrors the HuggingFace torch reference:
//!
//!   1. qkv/z via W4A16 GEMMs, a/b via BF16 GEMMs
//!   2. causal depthwise conv1d + SiLU on [query|key|value]
//!   3. split into query/key [seq, 2048] and value [seq, 6144]
//!   4. beta/g gates + L2-normed, head-repeated query/key
//!   5. serial SSM recurrence over the persistent f32 state
//!   6. RMSNorm(head_dim) × SiLU(z) output gate
//!   7. output projection back to the hidden width
//!
//! Both persistent states (`ssm_state`, `conv_state`) are caller-owned and
//! passed in; this module never allocates persistent device memory.

use apxinf_core::{DType, Error, Result, Shape, Tensor};

use super::contracts::{gpu_ptr, make_gpu_tensor, matrix_shape, require_buffers};
use super::gemm::{bf16, w4a16, W4A16WeightView};
use crate::buffer::CudaBuffer;
use crate::context::CudaContext;
use crate::ffi;
use crate::workspace::output_buffer;

/// Hidden width of the Qwen3.8-27B decoder layers.
pub const HIDDEN: usize = 5120;
/// Query/key width: 16 heads × 128.
pub const KEY_DIM: usize = 2048;
/// Value width: 48 heads × 128.
pub const VALUE_DIM: usize = 6144;
/// Query/key head count (repeated up to the value head count).
pub const NUM_HEADS: usize = 16;
/// Value head count.
pub const NUM_V_HEADS: usize = 48;
/// Per-head dimension.
pub const HEAD_DIM: usize = 128;
/// Causal depthwise conv kernel size.
pub const CONV_KERNEL: usize = 4;
/// conv_state holds the last kernel-1 conv inputs per channel.
pub const CONV_STATE: usize = CONV_KERNEL - 1;
/// Query/key repeat factor to reach the value head count.
pub const HEAD_REPEAT: usize = NUM_V_HEADS / NUM_HEADS;
/// Fused [query|key|value] conv width.
pub const CONV_CHANNELS: usize = KEY_DIM + KEY_DIM + VALUE_DIM;
/// Output-gate RMSNorm epsilon.
pub const GATE_EPS: f32 = 1e-6;

fn bf16_bytes(elements: usize, operation: &str) -> Result<usize> {
    elements
        .checked_mul(DType::BF16.size_in_bytes())
        .ok_or_else(|| Error::Other(format!("{operation} byte size overflow")))
}

pub enum OutProj<'a> {
    Quantized(W4A16WeightView<'a>),
    Dense(&'a Tensor),
}

/// One full Gated DeltaNet layer forward.
///
/// * `x`             — `[seq, 5120]` BF16 hidden states
/// * `in_proj_qkv`   — W4A16 view of `Linear(5120 → 10240)`
/// * `in_proj_z`     — W4A16 view of `Linear(5120 → 6144)`
/// * `a_weight`/`b_weight` — `[5120, 48]` BF16 (transposed to `[in, out]`)
/// * `a_log`/`dt_bias`     — `[48]` BF16 log-decay base / dt bias
/// * `conv1d`        — `[10240, 1, 4]` BF16 depthwise conv weight
/// * `norm_w`        — `[128]` BF16 RMSNorm weight
/// * `out_proj`      — `[6144, 5120]` BF16 (transposed to `[in, out]`)
/// * `ssm_state`     — persistent f32 `[48, 128, 128]` recurrence state
/// * `conv_state`    — persistent BF16 `[10240, 3]` conv history
///
/// Returns `[seq, 5120]` BF16. The `ssm_state`/`conv_state` buffers are
/// updated in place so consecutive decode steps accumulate state.
#[allow(clippy::too_many_arguments)]
pub fn forward(
    ctx: &CudaContext,
    x: &Tensor,
    in_proj_qkv: OutProj<'_>,
    in_proj_z: OutProj<'_>,
    a_weight: &Tensor,
    b_weight: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
    conv1d: &Tensor,
    norm_w: &Tensor,
    out_proj: OutProj<'_>,
    ssm_state: &CudaBuffer,
    conv_state: &CudaBuffer,
    seq_len: usize,
) -> Result<Tensor> {
    let (rows, cols) = matrix_shape(x, "gated delta net")?;
    if rows != seq_len || cols != HIDDEN {
        return Err(Error::Other(format!(
            "gated delta net expects [{seq_len}, {HIDDEN}] input, got [{rows}, {cols}]"
        )));
    }
    if x.dtype() != DType::BF16 {
        return Err(Error::Other(format!(
            "gated delta net expects BF16 input, got {}",
            x.dtype()
        )));
    }
    let seq = rows;

    // Static weight/layout validation before any device work.
    let weight_ok = a_weight.dtype() == DType::BF16
        && b_weight.dtype() == DType::BF16
        && a_log.dtype() == DType::BF16
        && dt_bias.dtype() == DType::BF16
        && conv1d.dtype() == DType::BF16
        && norm_w.dtype() == DType::BF16
        && a_weight.shape().dims() == [HIDDEN, NUM_V_HEADS]
        && b_weight.shape().dims() == [HIDDEN, NUM_V_HEADS]
        && a_log.shape().dims() == [NUM_V_HEADS]
        && dt_bias.shape().dims() == [NUM_V_HEADS]
        && conv1d.shape().dims() == [CONV_CHANNELS, 1, CONV_KERNEL]
        && norm_w.shape().dims() == [HEAD_DIM];
    if !weight_ok {
        return Err(Error::Other(
            "gated delta net weight shape/dtype mismatch (expected BF16 a/b [5120, 48], \
             a_log/dt_bias [48], conv1d [10240, 1, 4], norm_w [128], out_proj [6144, 5120])"
                .into(),
        ));
    }

    // Persistent state buffers: f32 SSM state and BF16 conv history.
    require_buffers(
        ctx,
        "gated delta net",
        &[
            (
                "ssm_state",
                ssm_state,
                NUM_V_HEADS * HEAD_DIM * HEAD_DIM * std::mem::size_of::<f32>(),
            ),
            (
                "conv_state",
                conv_state,
                bf16_bytes(CONV_CHANNELS * CONV_STATE, "gated delta net")?,
            ),
        ],
    )?;

    // ── Projections ────────────────────────────────────────────────────────
    let qkv = match in_proj_qkv {
        OutProj::Quantized(weight) => w4a16(ctx, x, weight),
        OutProj::Dense(weight) => bf16(ctx, x, weight),
    }?; // [seq, 10240]
    let z = match in_proj_z {
        OutProj::Quantized(weight) => w4a16(ctx, x, weight),
        OutProj::Dense(weight) => bf16(ctx, x, weight),
    }?; //     [seq, 6144]
    let a = bf16(ctx, x, a_weight)?; //        [seq, 48]
    let b = bf16(ctx, x, b_weight)?; //        [seq, 48]

    // ── Causal depthwise conv1d + SiLU on [query|key|value] ────────────────
    let conv_out = output_buffer(
        ctx,
        bf16_bytes(seq * CONV_CHANNELS, "gated delta net")?,
    )?;
    let qkv_buffer = CudaBuffer::from_tensor(&qkv).map_err(Error::Cuda)?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_delta_net_conv1d(
            qkv_buffer.ptr(),
            gpu_ptr(conv1d)?,
            conv_state.ptr(),
            conv_out.ptr(),
            seq as i32,
            CONV_CHANNELS as i32,
            CONV_KERNEL as i32,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }

    // ── Split 2048 | 2048 | 6144 ───────────────────────────────────────────
    let q = output_buffer(ctx, bf16_bytes(seq * KEY_DIM, "gated delta net")?)?;
    let k = output_buffer(ctx, bf16_bytes(seq * KEY_DIM, "gated delta net")?)?;
    let v = output_buffer(ctx, bf16_bytes(seq * VALUE_DIM, "gated delta net")?)?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_delta_net_split_qkv(
            conv_out.ptr(),
            q.ptr(),
            k.ptr(),
            v.ptr(),
            seq as i32,
            KEY_DIM as i32,
            KEY_DIM as i32,
            VALUE_DIM as i32,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }

    // ── beta/g gates, L2-normalized + head-repeated q/k ────────────────────
    // beta/g stay f32 (per-head scalars per token) to match the HF reference
    // precision; q_r/k_r are the BF16 L2-normalized repeat-interleaved heads.
    let beta = output_buffer(ctx, seq * NUM_V_HEADS * std::mem::size_of::<f32>())?;
    let g = output_buffer(ctx, seq * NUM_V_HEADS * std::mem::size_of::<f32>())?;
    let q_r = output_buffer(ctx, seq * VALUE_DIM * std::mem::size_of::<f32>())?;
    let k_r = output_buffer(ctx, seq * VALUE_DIM * std::mem::size_of::<f32>())?;
    let scale = (HEAD_DIM as f32).sqrt().recip();
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_delta_net_prepare(
            q.ptr(),
            k.ptr(),
            gpu_ptr(&a)?,
            gpu_ptr(&b)?,
            gpu_ptr(a_log)?,
            gpu_ptr(dt_bias)?,
            beta.ptr(),
            g.ptr(),
            q_r.ptr(),
            k_r.ptr(),
            seq as i32,
            NUM_HEADS as i32,
            NUM_V_HEADS as i32,
            HEAD_DIM as i32,
            scale,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }

    // ── SSM recurrence (serial over tokens, one block per value head) ──────
    let out = output_buffer(ctx, bf16_bytes(seq * VALUE_DIM, "gated delta net")?)?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_delta_net_step(
            q_r.ptr(),
            k_r.ptr(),
            v.ptr(),
            beta.ptr(),
            g.ptr(),
            ssm_state.ptr(),
            out.ptr(),
            seq as i32,
            NUM_V_HEADS as i32,
            HEAD_DIM as i32,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }

    // ── Output gate: RMSNorm(head_dim) × SiLU(z) ───────────────────────────
    let gated = output_buffer(ctx, bf16_bytes(seq * VALUE_DIM, "gated delta net")?)?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_delta_net_norm_gate(
            out.ptr(),
            gpu_ptr(&z)?,
            gpu_ptr(norm_w)?,
            gated.ptr(),
            seq as i32,
            NUM_V_HEADS as i32,
            HEAD_DIM as i32,
            GATE_EPS,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }

    // ── Output projection [seq, 6144] @ [6144, 5120] → [seq, 5120] ────────
    let gated_tensor = make_gpu_tensor(
        Shape::new(vec![seq, VALUE_DIM]),
        DType::BF16,
        ctx.device_id(),
        gated,
    );
    match out_proj {
        OutProj::Quantized(weight) => w4a16(ctx, &gated_tensor, weight),
        OutProj::Dense(weight) => bf16(ctx, &gated_tensor, weight),
    }
}
