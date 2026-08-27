//! Raw bindings for Qwen3.8-specific helper kernels.

use std::ffi::c_void;

use super::cuda::{cudaError_t, cudaStream_t};

extern "C" {
    pub fn apxinf_qwen35_sigmoid_bf16(
        input: *const c_void,
        output: *mut c_void,
        count: i64,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_qwen35_rope_partial_bf16(
        input: *const c_void,
        output: *mut c_void,
        head_dim: u32,
        n_heads: u32,
        seq_len: u32,
        rope_theta: f32,
        pos_offset: u32,
        rotary_dim: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_qwen35_split_q_gate_bf16(
        input: *const c_void,
        query: *mut c_void,
        gate: *mut c_void,
        rows: i32,
        n_heads: i32,
        head_dim: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_qwen35_slice_cols_bf16(
        input: *const c_void,
        output: *mut c_void,
        rows: i32,
        src_cols: i32,
        dst_cols: i32,
        col_off: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
}
