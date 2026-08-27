//! Qwen3.8 weight structures + HF weight-map loader.
//!
//! Quantized linear layers use the compressed-tensors "pack-quantized" W4A16
//! group-32 asymmetric layout:
//!   - weight_packed:  I32 [out, in/8]   (8 int4 per i32, low nibble first)
//!   - weight_scale:   BF16 [out, in/32]
//!   - weight_zero_point: I32 [out/8, in/32] (8 int8 per i32, packed along out)
//!   - weight_shape:   I64 [2] = [out, in]
//! dequant: w[o,i] = (nibble(o,i) - 8 - zp[o, i/32]) * scale[o, i/32]

use std::collections::HashMap;

use apxinf_core::{Error, Result, Tensor};

use super::config::Qwen35Config;

/// A quantized (W4A16) linear weight in HF `[out, in]` layout.
#[derive(Clone)]
pub struct QuantizedLinear {
    pub packed: Tensor,
    pub scale: Tensor,
    pub zero_point: Tensor,
    pub out_dim: usize,
    pub in_dim: usize,
}

impl QuantizedLinear {
    pub fn from_map(tensors: &mut HashMap<String, Tensor>, prefix: &str) -> Result<Self> {
        let packed = tensors
            .remove(&format!("{prefix}.weight_packed"))
            .ok_or_else(|| Error::Other(format!("missing {prefix}.weight_packed")))?;
        let scale = tensors
            .remove(&format!("{prefix}.weight_scale"))
            .ok_or_else(|| Error::Other(format!("missing {prefix}.weight_scale")))?;
        let zero_point = tensors
            .remove(&format!("{prefix}.weight_zero_point"))
            .ok_or_else(|| Error::Other(format!("missing {prefix}.weight_zero_point")))?;
        let shape = tensors
            .remove(&format!("{prefix}.weight_shape"))
            .ok_or_else(|| Error::Other(format!("missing {prefix}.weight_shape")))?;
        let shape_bytes = shape.as_raw_bytes()?;
        let dims = shape_bytes
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()) as usize)
            .collect::<Vec<_>>();
        if dims.len() != 2 {
            return Err(Error::Other(format!(
                "{prefix}.weight_shape has {} dims, expected 2",
                dims.len()
            )));
        }
        Ok(Self {
            packed,
            scale,
            zero_point,
            out_dim: dims[0],
            in_dim: dims[1],
        })
    }
}

/// Full-attention layer weights (16 layers).
pub struct FullAttnWeights {
    pub input_norm: Tensor,
    pub q_norm: Tensor,
    pub k_norm: Tensor,
    pub q_proj: QuantizedLinear,
    pub k_proj: QuantizedLinear,
    pub v_proj: QuantizedLinear,
    pub o_proj: QuantizedLinear,
    pub post_norm: Tensor,
    pub gate_proj: QuantizedLinear,
    pub up_proj: QuantizedLinear,
    pub down_proj: QuantizedLinear,
}

pub enum LinearOutProj {
    Quantized(QuantizedLinear),
    Dense(Tensor),
}

/// Linear-attention (Gated DeltaNet) layer weights (48 layers).
pub struct LinearAttnWeights {
    pub input_norm: Tensor,
    pub a_log: Tensor,
    pub conv1d: Tensor,
    pub dt_bias: Tensor,
    pub in_proj_a: Tensor,
    pub in_proj_b: Tensor,
    pub in_proj_qkv: QuantizedLinear,
    pub in_proj_z: QuantizedLinear,
    pub norm: Tensor,
    pub out_proj: LinearOutProj,
    pub post_norm: Tensor,
    pub gate_proj: QuantizedLinear,
    pub up_proj: QuantizedLinear,
    pub down_proj: QuantizedLinear,
}

pub enum Qwen35Layer {
    Full(FullAttnWeights),
    Linear(LinearAttnWeights),
}

pub struct Qwen35Weights {
    pub token_embedding: Tensor,
    pub layers: Vec<Qwen35Layer>,
    pub final_norm: Tensor,
    pub lm_head: Tensor,
}

impl Qwen35Weights {
    pub fn from_map(config: &Qwen35Config, mut tensors: HashMap<String, Tensor>) -> Result<Self> {
        let mut layers = Vec::with_capacity(config.n_layers);
        for i in 0..config.n_layers {
            let prefix = format!("model.language_model.layers.{i}");
            if config.is_full_attention(i) {
                layers.push(Qwen35Layer::Full(FullAttnWeights {
                    input_norm: take(&mut tensors, &format!("{prefix}.input_layernorm.weight"))?,
                    q_norm: take(&mut tensors, &format!("{prefix}.self_attn.q_norm.weight"))?,
                    k_norm: take(&mut tensors, &format!("{prefix}.self_attn.k_norm.weight"))?,
                    q_proj: QuantizedLinear::from_map(
                        &mut tensors,
                        &format!("{prefix}.self_attn.q_proj"),
                    )?,
                    k_proj: QuantizedLinear::from_map(
                        &mut tensors,
                        &format!("{prefix}.self_attn.k_proj"),
                    )?,
                    v_proj: QuantizedLinear::from_map(
                        &mut tensors,
                        &format!("{prefix}.self_attn.v_proj"),
                    )?,
                    o_proj: QuantizedLinear::from_map(
                        &mut tensors,
                        &format!("{prefix}.self_attn.o_proj"),
                    )?,
                    post_norm: take(
                        &mut tensors,
                        &format!("{prefix}.post_attention_layernorm.weight"),
                    )?,
                    gate_proj: QuantizedLinear::from_map(
                        &mut tensors,
                        &format!("{prefix}.mlp.gate_proj"),
                    )?,
                    up_proj: QuantizedLinear::from_map(
                        &mut tensors,
                        &format!("{prefix}.mlp.up_proj"),
                    )?,
                    down_proj: QuantizedLinear::from_map(
                        &mut tensors,
                        &format!("{prefix}.mlp.down_proj"),
                    )?,
                }));
            } else {
                layers.push(Qwen35Layer::Linear(LinearAttnWeights {
                    input_norm: take(&mut tensors, &format!("{prefix}.input_layernorm.weight"))?,
                    a_log: take(&mut tensors, &format!("{prefix}.linear_attn.A_log"))?,
                    conv1d: take(&mut tensors, &format!("{prefix}.linear_attn.conv1d.weight"))?,
                    dt_bias: take(&mut tensors, &format!("{prefix}.linear_attn.dt_bias"))?,
                    in_proj_a: take(&mut tensors, &format!("{prefix}.linear_attn.in_proj_a.weight"))?,
                    in_proj_b: take(&mut tensors, &format!("{prefix}.linear_attn.in_proj_b.weight"))?,
                    in_proj_qkv: QuantizedLinear::from_map(
                        &mut tensors,
                        &format!("{prefix}.linear_attn.in_proj_qkv"),
                    )?,
                    in_proj_z: QuantizedLinear::from_map(
                        &mut tensors,
                        &format!("{prefix}.linear_attn.in_proj_z"),
                    )?,
                    norm: take(&mut tensors, &format!("{prefix}.linear_attn.norm.weight"))?,
                    out_proj: {
                        let dense_key = format!("{prefix}.linear_attn.out_proj.weight");
                        if tensors.contains_key(&dense_key) {
                            // Layer 0's canonical checkpoint projection is dense.
                            // Ignore the optional non-canonical quantized sidecar.
                            LinearOutProj::Dense(take(&mut tensors, &dense_key)?)
                        } else {
                            LinearOutProj::Quantized(QuantizedLinear::from_map(
                                &mut tensors,
                                &format!("{prefix}.linear_attn.out_proj"),
                            )?)
                        }
                    },
                    post_norm: take(
                        &mut tensors,
                        &format!("{prefix}.post_attention_layernorm.weight"),
                    )?,
                    gate_proj: QuantizedLinear::from_map(
                        &mut tensors,
                        &format!("{prefix}.mlp.gate_proj"),
                    )?,
                    up_proj: QuantizedLinear::from_map(
                        &mut tensors,
                        &format!("{prefix}.mlp.up_proj"),
                    )?,
                    down_proj: QuantizedLinear::from_map(
                        &mut tensors,
                        &format!("{prefix}.mlp.down_proj"),
                    )?,
                }));
            }
        }

        let token_embedding =
            take(&mut tensors, "model.language_model.embed_tokens.weight")?;
        let final_norm = take(&mut tensors, "model.language_model.norm.weight")?;
        let lm_head = take(&mut tensors, "lm_head.weight")?;

        Ok(Self {
            token_embedding,
            layers,
            final_norm,
            lm_head,
        })
    }
}

fn take(tensors: &mut HashMap<String, Tensor>, name: &str) -> Result<Tensor> {
    tensors
        .remove(name)
        .ok_or_else(|| Error::Other(format!("missing tensor {name}")))
}
