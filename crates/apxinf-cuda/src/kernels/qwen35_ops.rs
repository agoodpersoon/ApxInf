//! Small Qwen3.8-specific helpers: sigmoid, partial RoPE, column slice.

use apxinf_core::{DType, Device, Error, Result, Shape, Tensor};

use crate::buffer::CudaBuffer;
use crate::context::CudaContext;
use crate::ffi;

/// Elementwise sigmoid (BF16 in/out, f32 compute).
pub fn sigmoid(ctx: &CudaContext, input: &Tensor) -> Result<Tensor> {
    if input.dtype() != DType::BF16 || input.device() != Device::Cuda(ctx.device_id()) {
        return Err(Error::Other(format!(
            "sigmoid expects BF16 on CUDA {}, got {} on {}",
            ctx.device_id(),
            input.dtype(),
            input.device()
        )));
    }
    let count = input.shape().numel();
    let output = crate::workspace::output_buffer(ctx, count * DType::BF16.size_in_bytes())?;
    let input_buf = CudaBuffer::from_tensor(input).map_err(Error::Cuda)?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_qwen35_sigmoid_bf16(
            input_buf.ptr(),
            output.ptr(),
            count as i64,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(output.into_tensor(input.shape().clone(), DType::BF16))
}

/// Partial RoPE: rotate the first `rotary_dim` of `head_dim`, pass the rest.
/// Input `[seq, n_heads, head_dim]` BF16.
pub fn rope_partial(
    ctx: &CudaContext,
    input: &Tensor,
    n_heads: usize,
    head_dim: usize,
    rope_theta: f32,
    pos_offset: u32,
    rotary_dim: usize,
) -> Result<Tensor> {
    if input.dtype() != DType::BF16 || input.device() != Device::Cuda(ctx.device_id()) {
        return Err(Error::Other("rope_partial expects BF16 on the active CUDA device".into()));
    }
    let dims = input.shape().dims();
    if dims.len() != 3 || dims[1] != n_heads || dims[2] != head_dim {
        return Err(Error::Other(format!(
            "rope_partial expects [seq,{n_heads},{head_dim}], got {dims:?}"
        )));
    }
    let seq_len = dims[0];
    let output = crate::workspace::output_buffer(ctx, input.size_in_bytes())?;
    let input_buf = CudaBuffer::from_tensor(input).map_err(Error::Cuda)?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_qwen35_rope_partial_bf16(
            input_buf.ptr(),
            output.ptr(),
            head_dim as u32,
            n_heads as u32,
            seq_len as u32,
            rope_theta,
            pos_offset,
            rotary_dim as u32,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(output.into_tensor(input.shape().clone(), DType::BF16))
}

/// Split HF's interleaved q_proj output into query and gate tensors.
pub fn split_q_gate(
    ctx: &CudaContext,
    input: &Tensor,
    n_heads: usize,
    head_dim: usize,
) -> Result<(Tensor, Tensor)> {
    if input.dtype() != DType::BF16 || input.device() != Device::Cuda(ctx.device_id()) {
        return Err(Error::Other("split_q_gate expects BF16 on the active CUDA device".into()));
    }
    let dims = input.shape().dims();
    if dims.len() != 2 || dims[1] != 2 * n_heads * head_dim {
        return Err(Error::Other(format!(
            "split_q_gate expects [rows,{}], got {dims:?}",
            2 * n_heads * head_dim
        )));
    }
    let rows = dims[0];
    let bytes = rows * n_heads * head_dim * DType::BF16.size_in_bytes();
    let query = crate::workspace::output_buffer(ctx, bytes)?;
    let gate = crate::workspace::output_buffer(ctx, bytes)?;
    let input_buf = CudaBuffer::from_tensor(input).map_err(Error::Cuda)?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_qwen35_split_q_gate_bf16(
            input_buf.ptr(),
            query.ptr(),
            gate.ptr(),
            rows as i32,
            n_heads as i32,
            head_dim as i32,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    let shape = Shape::new(vec![rows, n_heads * head_dim]);
    Ok((
        query.into_tensor(shape.clone(), DType::BF16),
        gate.into_tensor(shape, DType::BF16),
    ))
}

/// Copy a column range: `output[r,c] = input[r, col_off + c]` for `[rows, src_cols]` -> `[rows, dst_cols]`.
pub fn slice_cols(
    ctx: &CudaContext,
    input: &Tensor,
    col_off: usize,
    dst_cols: usize,
) -> Result<Tensor> {
    if input.dtype() != DType::BF16 || input.device() != Device::Cuda(ctx.device_id()) {
        return Err(Error::Other("slice_cols expects BF16 on the active CUDA device".into()));
    }
    let dims = input.shape().dims();
    if dims.len() != 2 {
        return Err(Error::Other(format!("slice_cols expects 2D input, got {dims:?}")));
    }
    let (rows, src_cols) = (dims[0], dims[1]);
    if col_off + dst_cols > src_cols {
        return Err(Error::Other(format!(
            "slice_cols range [{col_off}, {}) exceeds src_cols {src_cols}",
            col_off + dst_cols
        )));
    }
    let output = crate::workspace::output_buffer(ctx, rows * dst_cols * DType::BF16.size_in_bytes())?;
    let input_buf = CudaBuffer::from_tensor(input).map_err(Error::Cuda)?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_qwen35_slice_cols_bf16(
            input_buf.ptr(),
            output.ptr(),
            rows as i32,
            src_cols as i32,
            dst_cols as i32,
            col_off as i32,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(output.into_tensor(Shape::new(vec![rows, dst_cols]), DType::BF16))
}
