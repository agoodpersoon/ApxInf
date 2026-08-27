//! W4A16 group-32 asymmetric quantized GEMM (compressed-tensors pack-quantized).

use apxinf_core::{DType, Device, Error, Result, Shape, Tensor};

use crate::buffer::CudaBuffer;
use crate::context::CudaContext;
use crate::ffi;

use super::bf16::gemm_bf16;

/// Borrowed W4A16 weight view. The weight stays in HF `[out, in]` packed layout.
#[derive(Clone, Copy)]
pub struct W4A16WeightView<'a> {
    /// int32 [out, in/8] — 8 int4 per i32, low nibble first.
    pub packed: &'a Tensor,
    /// bf16 [out, in/32] per-group scale.
    pub scale: &'a Tensor,
    /// int32 [out/8, in/32] packed int8 zero-points (along out).
    pub zero_point: &'a Tensor,
    pub output_dim: usize,
    pub input_dim: usize,
    pub group_size: usize,
}

/// W4A16 GEMM: `C[m,n] = sum_k A[m,k] * ((q(n,k) - zp(n,k/gs)) * s(n,k/gs))`.
pub fn gemm_w4a16(
    ctx: &CudaContext,
    activation: &Tensor,
    weight: W4A16WeightView<'_>,
) -> Result<Tensor> {
    if activation.dtype() != DType::BF16 || activation.device() != Device::Cuda(ctx.device_id()) {
        return Err(Error::Other(format!(
            "gemm_w4a16 expects a BF16 activation on CUDA {}, got {} on {}",
            ctx.device_id(),
            activation.dtype(),
            activation.device()
        )));
    }
    let dims = activation.shape().dims();
    if dims.len() != 2 || dims[1] != weight.input_dim {
        return Err(Error::Other(format!(
            "gemm_w4a16 activation shape mismatch: expected [M,{}], got {dims:?}",
            weight.input_dim
        )));
    }
    if weight.group_size != 32 {
        return Err(Error::Other(format!(
            "gemm_w4a16 only supports group_size 32, got {}",
            weight.group_size
        )));
    }
    if weight.packed.dtype() != DType::I32
        || weight.scale.dtype() != DType::BF16
        || weight.zero_point.dtype() != DType::I32
    {
        return Err(Error::Other(
            "gemm_w4a16 weight dtypes must be I32 (packed/zp) and BF16 (scale)".into(),
        ));
    }

    let m = dims[0];
    let n = weight.output_dim;
    let k = weight.input_dim;

    // The fused W4 kernel is useful for decode, but its scalar dequantization
    // and FP32 accumulation leave Tensor Cores idle on large prefills. For a
    // sufficiently wide activation, materialize the BF16 weight once and let
    // cuBLAS use the native BF16 Tensor Core path. Decode (M=1) keeps the
    // fused kernel to avoid the full dense weight allocation.
    if m >= 128 {
        let dense = dequant_w4a16_bf16(
            ctx,
            weight.packed,
            weight.scale,
            weight.zero_point,
            n,
            k,
        )?;
        return gemm_bf16(ctx, activation, &dense);
    }

    let activation = CudaBuffer::from_tensor(activation).map_err(Error::Cuda)?;
    let packed = CudaBuffer::from_tensor(weight.packed).map_err(Error::Cuda)?;
    let scale = CudaBuffer::from_tensor(weight.scale).map_err(Error::Cuda)?;
    let zero_point = CudaBuffer::from_tensor(weight.zero_point).map_err(Error::Cuda)?;
    let output = crate::workspace::output_buffer(ctx, m * n * DType::BF16.size_in_bytes())?;

    unsafe {
        ffi::check_cuda(ffi::apxinf_static_w4a16_gemm_bf16(
            activation.ptr(),
            packed.ptr(),
            scale.ptr(),
            zero_point.ptr(),
            output.ptr(),
            m as i32,
            n as i32,
            k as i32,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(output.into_tensor(Shape::new(vec![m, n]), DType::BF16))
}

pub fn dequant_w4a16_bf16(
    ctx: &CudaContext,
    packed_tensor: &Tensor,
    scale_tensor: &Tensor,
    zero_point_tensor: &Tensor,
    output_dim: usize,
    input_dim: usize,
) -> Result<Tensor> {
    if input_dim % 32 != 0 || output_dim % 8 != 0 {
        return Err(Error::Other(format!(
            "W4A16 dense dequant requires K divisible by 32 and N by 8, got K={} N={}",
            input_dim, output_dim
        )));
    }
    let packed = CudaBuffer::from_tensor(packed_tensor).map_err(Error::Cuda)?;
    let scale = CudaBuffer::from_tensor(scale_tensor).map_err(Error::Cuda)?;
    let zero_point = CudaBuffer::from_tensor(zero_point_tensor).map_err(Error::Cuda)?;
    let bytes = input_dim
        .checked_mul(output_dim)
        .and_then(|elements| elements.checked_mul(DType::BF16.size_in_bytes()))
        .ok_or_else(|| Error::Other("W4A16 dense dequant size overflow".into()))?;
    let output = crate::workspace::output_buffer(ctx, bytes)?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_w4a16_dequant_bf16(
            packed.ptr(),
            scale.ptr(),
            zero_point.ptr(),
            output.ptr(),
            output_dim as i32,
            input_dim as i32,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(output.into_tensor(
        Shape::new(vec![input_dim, output_dim]),
        DType::BF16,
    ))
}
