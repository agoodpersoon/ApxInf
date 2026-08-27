//! CUDA-only forward pass for Qwen3.8-27B hybrid model.
//!
//! Drives `apxinf_cuda` kernels directly (same pattern as `llama/model.rs`).
//! Full-attention layers keep a KV cache; linear-attention (Gated DeltaNet)
//! layers keep an SSM + conv recurrent state.

use std::sync::Arc;

use apxinf_core::{Backend, Device, Error, Result, Tensor};
use apxinf_loader::ModelConfig;

use crate::llm_trait::{LlmCapabilities, LlmInput, LlmTrait};
use crate::profiling::GenerationProfile;
use crate::qwen35::config::Qwen35Config;
use crate::qwen35::weights::{LinearOutProj, Qwen35Layer, Qwen35Weights, QuantizedLinear};

#[cfg(feature = "cuda")]
use crate::accelerator::cuda::{
    downcast_arc, kernels, transfers, CudaBackend, CudaBuffer, CudaKVCache,
};
use apxinf_core::KvCache;

#[cfg(feature = "cuda")]
use apxinf_cuda::kernels::gemm::W4A16WeightView;

#[cfg(feature = "cuda")]
struct FullDenseCache {
    q_proj: Tensor,
    k_proj: Tensor,
    v_proj: Tensor,
    o_proj: Tensor,
    gate_proj: Tensor,
    up_proj: Tensor,
    down_proj: Tensor,
}

#[cfg(feature = "cuda")]
struct LinearDenseCache {
    in_proj_qkv: Tensor,
    in_proj_z: Tensor,
    out_proj: Option<Tensor>,
    gate_proj: Tensor,
    up_proj: Tensor,
    down_proj: Tensor,
}

pub struct GeneralQwen35 {
    config: Qwen35Config,
    weights: Qwen35Weights,
    #[cfg(feature = "cuda")]
    backend: Arc<CudaBackend>,
    /// KV cache for the 16 full-attention layers only (slot = layer_idx / interval).
    #[cfg(feature = "cuda")]
    kv: CudaKVCache,
    /// SSM recurrent state [n_v_heads, k_dim, v_dim] f32 per linear layer.
    #[cfg(feature = "cuda")]
    ssm_state: Vec<CudaBuffer>,
    #[cfg(feature = "cuda")]
    conv_state: Vec<CudaBuffer>,
}

impl GeneralQwen35 {
    pub fn new(config: Qwen35Config, weights: Qwen35Weights, backend: Arc<dyn Backend>) -> Result<Self> {
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (config, weights, backend);
            return Err(Error::Other("qwen35 requires the cuda feature".into()));
        }
        #[cfg(feature = "cuda")]
        {
            let backend = downcast_arc(backend)
                .ok_or_else(|| Error::Other("qwen35 needs a CudaBackend".to_string()))?;
            let ctx = backend.context();

            let weights = transfer_weights(weights, ctx)?;

            let n_full = config.num_full_attention_layers();
            let kv = CudaKVCache::new(
                ctx.device_id(),
                n_full,
                config.n_kv_heads,
                config.head_dim,
                config.max_seq_len,
            )?;

            let n_linear = config.n_layers - n_full;
            let k_dim = config.linear_key_head_dim;
            let v_dim = config.linear_value_head_dim;
            let n_v_heads = config.linear_num_value_heads;
            let ssm_bytes = n_v_heads * k_dim * v_dim * 4; // f32
            let conv_channels = config.linear_key_dim() * 2 + config.linear_value_dim();
            let conv_bytes = conv_channels * (config.linear_conv_kernel_dim - 1) * 2; // bf16
            let mut ssm_state = Vec::with_capacity(n_linear);
            let mut conv_state = Vec::with_capacity(n_linear);
            for _ in 0..n_linear {
                ssm_state.push(CudaBuffer::alloc_zeros(ssm_bytes, ctx.device_id()).map_err(|e| Error::Cuda(e.to_string()))?);
                conv_state.push(CudaBuffer::alloc_zeros(conv_bytes, ctx.device_id()).map_err(|e| Error::Cuda(e.to_string()))?);
            }

            Ok(Self {
                config,
                weights,
                backend,
                kv,
                ssm_state,
                conv_state,
            })
        }
    }

    #[cfg(feature = "cuda")]
    fn forward_impl(&mut self, token_ids: &[u32], start_pos: u32) -> Result<Tensor> {
        let ctx = self.backend.context();
        let seq_len = token_ids.len();

        // Upload token ids.
        let ids_buf = CudaBuffer::alloc(seq_len * 4, ctx.device_id()).map_err(|e| Error::Cuda(e.to_string()))?;
        let host: Vec<u8> = token_ids.iter().flat_map(|id| id.to_ne_bytes()).collect();
        ids_buf.copy_from_host(&host).map_err(|e| Error::Cuda(e.to_string()))?;

        let mut x = kernels::embedding::lookup(
            ctx,
            &self.weights.token_embedding,
            &ids_buf,
            seq_len,
        )?;

        let mut linear_slot = 0usize;
        const LAYER_CHUNK: usize = 4096;
        for layer_idx in 0..self.config.n_layers {
            let slot = match self.weights.layers[layer_idx] {
                Qwen35Layer::Full(_) => layer_idx / self.config.full_attention_interval,
                Qwen35Layer::Linear(_) => {
                    let value = linear_slot;
                    linear_slot += 1;
                    value
                }
            };
            // Long prefills are split into chunks to keep temporary activations
            // bounded. Materialize this layer's W4 weights once so each chunk
            // can use BF16 Tensor-Core GEMM instead of repeating dequantization.
            let full_dense = if seq_len > LAYER_CHUNK {
                match &self.weights.layers[layer_idx] {
                    Qwen35Layer::Full(layer) => Some(self.prepare_full_dense(ctx, layer)?),
                    Qwen35Layer::Linear(_) => None,
                }
            } else {
                None
            };
            let linear_dense = if seq_len > LAYER_CHUNK {
                match &self.weights.layers[layer_idx] {
                    Qwen35Layer::Linear(layer) => Some(self.prepare_linear_dense(ctx, layer)?),
                    Qwen35Layer::Full(_) => None,
                }
            } else {
                None
            };
            let base_cache_pos = if start_pos > 0 { self.kv.seq_len() as usize } else { 0 };
            let mut layer_out: Option<Tensor> = None;
            for chunk_start in (0..seq_len).step_by(LAYER_CHUNK) {
                let chunk_len = (seq_len - chunk_start).min(LAYER_CHUNK);
                let x_chunk = kernels::elementwise::slice_rows_bf16(ctx, &x, chunk_start, chunk_len)?;
                let y = match &self.weights.layers[layer_idx] {
                    Qwen35Layer::Full(_) => self.forward_full(
                        ctx, layer_idx, &x_chunk, start_pos + chunk_start as u32,
                        chunk_len, slot, base_cache_pos + chunk_start, seq_len, full_dense.as_ref(),
                    )?,
                    Qwen35Layer::Linear(_) => self.forward_linear(
                        ctx, layer_idx, &x_chunk, chunk_len, slot, linear_dense.as_ref(),
                    )?,
                };
                layer_out = Some(match layer_out {
                    Some(previous) => kernels::elementwise::concat_rows_bf16(ctx, &previous, &y)?,
                    None => y,
                });
            }
            x = layer_out.ok_or_else(|| Error::Other("empty decoder layer output".into()))?;
        }

        // Advance the shared KV position only after every layer has processed this
        // request, so the next decode step sees the complete prefill.
        self.kv.advance(seq_len);

        let normed = kernels::norm::rms_one_plus_bf16(ctx, &x, &self.weights.final_norm, self.config.rms_norm_eps)?;
        let logits = kernels::gemm::gemm_bf16_last_row(ctx, &normed, &self.weights.lm_head)?;
        self.backend.to_cpu(&logits)
    }

    #[cfg(feature = "cuda")]
    fn forward_full(
        &self,
        ctx: &apxinf_cuda::CudaContext,
        layer_idx: usize,
        x: &Tensor,
        start_pos: u32,
        seq_len: usize,
        kv_slot: usize,
        cache_pos: usize,
        request_seq_len: usize,
        dense: Option<&FullDenseCache>,
    ) -> Result<Tensor> {
        let cfg = &self.config;
        let layer = match &self.weights.layers[layer_idx] {
            Qwen35Layer::Full(l) => l,
            _ => unreachable!(),
        };
        let residual = x.clone();

        let normed = kernels::norm::rms_one_plus_bf16(ctx, x, &layer.input_norm, cfg.rms_norm_eps)?;

        // QKV projections. q_proj outputs [heads*head_dim*2] = query + gate.
        let qkv = match dense {
            Some(dense) => kernels::gemm::bf16(ctx, &normed, &dense.q_proj),
            None => kernels::gemm::w4a16(ctx, &normed, view(&layer.q_proj)),
        }?;
        let k = match dense {
            Some(dense) => kernels::gemm::bf16(ctx, &normed, &dense.k_proj),
            None => kernels::gemm::w4a16(ctx, &normed, view(&layer.k_proj)),
        }?;
        let v = match dense {
            Some(dense) => kernels::gemm::bf16(ctx, &normed, &dense.v_proj),
            None => kernels::gemm::w4a16(ctx, &normed, view(&layer.v_proj)),
        }?;

        let (q, gate) =
            kernels::qwen35_ops::split_q_gate(ctx, &qkv, cfg.n_heads, cfg.head_dim)?;

        let q = q.reshape(vec![seq_len, cfg.n_heads, cfg.head_dim])?;
        let gate = gate.reshape(vec![seq_len, cfg.n_heads, cfg.head_dim])?;
        let k = k.reshape(vec![seq_len, cfg.n_kv_heads, cfg.head_dim])?;
        let v = v.reshape(vec![seq_len, cfg.n_kv_heads, cfg.head_dim])?;

        // Per-head RMSNorm on q and k (norm over head_dim).
        let q = rms_per_head(ctx, &q, &layer.q_norm, cfg.rms_norm_eps)?;
        let k = rms_per_head(ctx, &k, &layer.k_norm, cfg.rms_norm_eps)?;

        // Partial RoPE.
        let rotary_dim = cfg.rotary_dim();
        let q = kernels::qwen35_ops::rope_partial(
            ctx, &q, cfg.n_heads, cfg.head_dim, cfg.rope_theta, start_pos, rotary_dim,
        )?;
        let k = kernels::qwen35_ops::rope_partial(
            ctx, &k, cfg.n_kv_heads, cfg.head_dim, cfg.rope_theta, start_pos, rotary_dim,
        )?;

        // Append KV + attention.
        kernels::cache::append(
            ctx,
            self.kv.k_buffer(kv_slot),
            &k,
            cfg.n_kv_heads,
            cfg.head_dim,
            cfg.max_seq_len,
            cache_pos,
            seq_len,
        )?;
        kernels::cache::append(
            ctx,
            self.kv.v_buffer(kv_slot),
            &v,
            cfg.n_kv_heads,
            cfg.head_dim,
            cfg.max_seq_len,
            cache_pos,
            seq_len,
        )?;
        let kv_len = cache_pos + seq_len;
        let attn = kernels::attention::sdpa(
            ctx,
            &q,
            &self.kv,
            kv_slot,
            cfg.n_heads,
            cfg.n_kv_heads,
            cfg.head_dim,
            kv_len,
            cfg.max_seq_len,
            cache_pos as u32,
            request_seq_len,
        )?; // [seq, heads*head_dim]

        // Output gate: attn * sigmoid(gate).
        let gate2 = kernels::qwen35_ops::sigmoid(ctx, &gate)?;
        let gated = kernels::elementwise::mul(ctx, &attn, &gate2)?;
        let o = match dense {
            Some(dense) => kernels::gemm::bf16(ctx, &gated, &dense.o_proj),
            None => kernels::gemm::w4a16(ctx, &gated, view(&layer.o_proj)),
        }?;
        let mut out = kernels::elementwise::add(ctx, &residual, &o)?;

        // MLP.
        let residual2 = out.clone();
        let normed = kernels::norm::rms_one_plus_bf16(ctx, &out, &layer.post_norm, cfg.rms_norm_eps)?;
        let gate_up = match dense {
            Some(dense) => kernels::gemm::bf16(ctx, &normed, &dense.gate_proj),
            None => kernels::gemm::w4a16(ctx, &normed, view(&layer.gate_proj)),
        }?;
        let up = match dense {
            Some(dense) => kernels::gemm::bf16(ctx, &normed, &dense.up_proj),
            None => kernels::gemm::w4a16(ctx, &normed, view(&layer.up_proj)),
        }?;
        let act = kernels::activation::silu(ctx, &gate_up)?;
        let hidden = kernels::elementwise::mul(ctx, &act, &up)?;
        let down = match dense {
            Some(dense) => kernels::gemm::bf16(ctx, &hidden, &dense.down_proj),
            None => kernels::gemm::w4a16(ctx, &hidden, view(&layer.down_proj)),
        }?;
        out = kernels::elementwise::add(ctx, &residual2, &down)?;
        Ok(out)
    }

    #[cfg(feature = "cuda")]
    fn forward_linear(
        &self,
        ctx: &apxinf_cuda::CudaContext,
        layer_idx: usize,
        x: &Tensor,
        seq_len: usize,
        slot: usize,
        dense: Option<&LinearDenseCache>,
    ) -> Result<Tensor> {
        let cfg = &self.config;
        let layer = match &self.weights.layers[layer_idx] {
            Qwen35Layer::Linear(l) => l,
            _ => unreachable!(),
        };
        let residual = x.clone();
        let normed = kernels::norm::rms_one_plus_bf16(ctx, x, &layer.input_norm, cfg.rms_norm_eps)?;

        let out = kernels::linear_attn::forward(
            ctx,
            &normed,
            match dense {
                Some(dense) => kernels::linear_attn::OutProj::Dense(&dense.in_proj_qkv),
                None => kernels::linear_attn::OutProj::Quantized(view(&layer.in_proj_qkv)),
            },
            match dense {
                Some(dense) => kernels::linear_attn::OutProj::Dense(&dense.in_proj_z),
                None => kernels::linear_attn::OutProj::Quantized(view(&layer.in_proj_z)),
            },
            &layer.in_proj_a,
            &layer.in_proj_b,
            &layer.a_log,
            &layer.dt_bias,
            &layer.conv1d,
            &layer.norm,
            match dense.and_then(|dense| dense.out_proj.as_ref()) {
                Some(weight) => kernels::linear_attn::OutProj::Dense(weight),
                None => match &layer.out_proj {
                    LinearOutProj::Quantized(weight) => {
                        kernels::linear_attn::OutProj::Quantized(view(weight))
                    }
                    LinearOutProj::Dense(weight) => kernels::linear_attn::OutProj::Dense(weight),
                },
            },
            &self.ssm_state[slot],
            &self.conv_state[slot],
            seq_len,
        )?;

        let mut out = kernels::elementwise::add(ctx, &residual, &out)?;

        // MLP.
        let residual2 = out.clone();
        let normed = kernels::norm::rms_one_plus_bf16(ctx, &out, &layer.post_norm, cfg.rms_norm_eps)?;
        let gate_up = match dense {
            Some(dense) => kernels::gemm::bf16(ctx, &normed, &dense.gate_proj),
            None => kernels::gemm::w4a16(ctx, &normed, view(&layer.gate_proj)),
        }?;
        let up = match dense {
            Some(dense) => kernels::gemm::bf16(ctx, &normed, &dense.up_proj),
            None => kernels::gemm::w4a16(ctx, &normed, view(&layer.up_proj)),
        }?;
        let act = kernels::activation::silu(ctx, &gate_up)?;
        let hidden = kernels::elementwise::mul(ctx, &act, &up)?;
        let down = match dense {
            Some(dense) => kernels::gemm::bf16(ctx, &hidden, &dense.down_proj),
            None => kernels::gemm::w4a16(ctx, &hidden, view(&layer.down_proj)),
        }?;
        out = kernels::elementwise::add(ctx, &residual2, &down)?;
        Ok(out)
    }

    #[cfg(feature = "cuda")]
    fn prepare_full_dense(
        &self,
        ctx: &apxinf_cuda::CudaContext,
        layer: &crate::qwen35::weights::FullAttnWeights,
    ) -> Result<FullDenseCache> {
        Ok(FullDenseCache {
            q_proj: dequant(ctx, &layer.q_proj)?,
            k_proj: dequant(ctx, &layer.k_proj)?,
            v_proj: dequant(ctx, &layer.v_proj)?,
            o_proj: dequant(ctx, &layer.o_proj)?,
            gate_proj: dequant(ctx, &layer.gate_proj)?,
            up_proj: dequant(ctx, &layer.up_proj)?,
            down_proj: dequant(ctx, &layer.down_proj)?,
        })
    }

    #[cfg(feature = "cuda")]
    fn prepare_linear_dense(
        &self,
        ctx: &apxinf_cuda::CudaContext,
        layer: &crate::qwen35::weights::LinearAttnWeights,
    ) -> Result<LinearDenseCache> {
        Ok(LinearDenseCache {
            in_proj_qkv: dequant(ctx, &layer.in_proj_qkv)?,
            in_proj_z: dequant(ctx, &layer.in_proj_z)?,
            out_proj: match &layer.out_proj {
                LinearOutProj::Quantized(weight) => Some(dequant(ctx, weight)?),
                LinearOutProj::Dense(weight) => Some(weight.clone()),
            },
            gate_proj: dequant(ctx, &layer.gate_proj)?,
            up_proj: dequant(ctx, &layer.up_proj)?,
            down_proj: dequant(ctx, &layer.down_proj)?,
        })
    }
}

#[cfg(feature = "cuda")]
fn view(q: &QuantizedLinear) -> W4A16WeightView<'_> {
    W4A16WeightView {
        packed: &q.packed,
        scale: &q.scale,
        zero_point: &q.zero_point,
        output_dim: q.out_dim,
        input_dim: q.in_dim,
        group_size: 32,
    }
}

#[cfg(feature = "cuda")]
fn dequant(ctx: &apxinf_cuda::CudaContext, q: &QuantizedLinear) -> Result<Tensor> {
    let v = view(q);
    kernels::gemm::dequant_w4a16_bf16(
        ctx,
        v.packed,
        v.scale,
        v.zero_point,
        v.output_dim,
        v.input_dim,
    )
}

/// Per-head RMSNorm: normalize each (seq, head) row over head_dim.
#[cfg(feature = "cuda")]
fn rms_per_head(
    ctx: &apxinf_cuda::CudaContext,
    x: &Tensor,
    weight: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    let dims = x.shape().dims();
    let (seq, heads, head_dim) = (dims[0], dims[1], dims[2]);
    let flat = x.reshape(vec![seq * heads, head_dim])?;
    let normed = kernels::norm::rms_one_plus_bf16(ctx, &flat, weight, eps)?;
    normed.reshape(vec![seq, heads, head_dim])
}

#[cfg(feature = "cuda")]
fn transfer_weights(weights: Qwen35Weights, ctx: &apxinf_cuda::CudaContext) -> Result<Qwen35Weights> {
    let mut layers = Vec::with_capacity(weights.layers.len());
    for layer in weights.layers {
        layers.push(match layer {
            Qwen35Layer::Full(l) => Qwen35Layer::Full(crate::qwen35::weights::FullAttnWeights {
                input_norm: to_dev(&l.input_norm, ctx)?,
                q_norm: to_dev(&l.q_norm, ctx)?,
                k_norm: to_dev(&l.k_norm, ctx)?,
                q_proj: to_dev_q(&l.q_proj, ctx)?,
                k_proj: to_dev_q(&l.k_proj, ctx)?,
                v_proj: to_dev_q(&l.v_proj, ctx)?,
                o_proj: to_dev_q(&l.o_proj, ctx)?,
                post_norm: to_dev(&l.post_norm, ctx)?,
                gate_proj: to_dev_q(&l.gate_proj, ctx)?,
                up_proj: to_dev_q(&l.up_proj, ctx)?,
                down_proj: to_dev_q(&l.down_proj, ctx)?,
            }),
            Qwen35Layer::Linear(l) => Qwen35Layer::Linear(crate::qwen35::weights::LinearAttnWeights {
                input_norm: to_dev(&l.input_norm, ctx)?,
                a_log: to_dev(&l.a_log, ctx)?,
                conv1d: to_dev(&l.conv1d, ctx)?,
                dt_bias: to_dev(&l.dt_bias, ctx)?,
                in_proj_a: to_dev(&transpose_bf16(&l.in_proj_a)?, ctx)?,
                in_proj_b: to_dev(&transpose_bf16(&l.in_proj_b)?, ctx)?,
                in_proj_qkv: to_dev_q(&l.in_proj_qkv, ctx)?,
                in_proj_z: to_dev_q(&l.in_proj_z, ctx)?,
                norm: to_dev(&l.norm, ctx)?,
                out_proj: match l.out_proj {
                    LinearOutProj::Quantized(weight) => {
                        LinearOutProj::Quantized(to_dev_q(&weight, ctx)?)
                    }
                    LinearOutProj::Dense(weight) => {
                        LinearOutProj::Dense(to_dev(&transpose_bf16(&weight)?, ctx)?)
                    }
                },
                post_norm: to_dev(&l.post_norm, ctx)?,
                gate_proj: to_dev_q(&l.gate_proj, ctx)?,
                up_proj: to_dev_q(&l.up_proj, ctx)?,
                down_proj: to_dev_q(&l.down_proj, ctx)?,
            }),
        });
    }
    Ok(Qwen35Weights {
        token_embedding: to_dev(&weights.token_embedding, ctx)?,
        layers,
        final_norm: to_dev(&weights.final_norm, ctx)?,
        lm_head: to_dev(&transpose_bf16(&weights.lm_head)?, ctx)?,
    })
}

#[cfg(feature = "cuda")]
fn to_dev(t: &Tensor, ctx: &apxinf_cuda::CudaContext) -> Result<Tensor> {
    transfers::to_cuda(t, ctx.device_id()).map_err(|e| Error::Cuda(e.to_string()))
}

#[cfg(feature = "cuda")]
fn to_dev_q(q: &QuantizedLinear, ctx: &apxinf_cuda::CudaContext) -> Result<QuantizedLinear> {
    Ok(QuantizedLinear {
        packed: to_dev(&q.packed, ctx)?,
        scale: to_dev(&q.scale, ctx)?,
        zero_point: to_dev(&q.zero_point, ctx)?,
        out_dim: q.out_dim,
        in_dim: q.in_dim,
    })
}

/// Transpose a BF16 2D tensor [out, in] -> [in, out] on CPU (for cuBLAS NN GEMM).
#[cfg(feature = "cuda")]
fn transpose_bf16(t: &Tensor) -> Result<Tensor> {
    let dims = t.shape().dims();
    if dims.len() != 2 {
        return Err(Error::Other("transpose_bf16 expects 2D".into()));
    }
    let (rows, cols) = (dims[0], dims[1]);
    let data = t.as_bf16().map_err(|e| Error::Other(e.to_string()))?;
    let mut out = vec![half::bf16::from_f32(0.0); rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            out[j * rows + i] = data[i * cols + j];
        }
    }
    Tensor::from_bf16(vec![cols, rows], &out).map_err(Error::from)
}

impl LlmTrait for GeneralQwen35 {
    fn load(_config: ModelConfig, _weights: std::collections::HashMap<String, Tensor>, _device: Device) -> Result<Self>
    where
        Self: Sized,
    {
        Err(Error::Other(
            "GeneralQwen35 must be loaded via qwen35::load_qwen35".into(),
        ))
    }

    fn forward(&mut self, token_ids: &[u32], start_pos: u32) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        {
            if start_pos == 0 {
                let _ = self.kv.clear();
                for b in self.ssm_state.iter() {
                    let _ = b.zero();
                }
                for b in self.conv_state.iter() {
                    let _ = b.zero();
                }
            }
            self.forward_impl(token_ids, start_pos)
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (token_ids, start_pos);
            Err(Error::Other("qwen35 requires cuda".into()))
        }
    }

    fn capabilities(&self) -> LlmCapabilities {
        LlmCapabilities::TEXT_ONLY
    }

    fn reset(&mut self) {
        #[cfg(feature = "cuda")]
        {
            let _ = self.kv.clear();
            for b in self.ssm_state.iter() {
                let _ = b.zero();
            }
            for b in self.conv_state.iter() {
                let _ = b.zero();
            }
        }
    }

    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }
}
