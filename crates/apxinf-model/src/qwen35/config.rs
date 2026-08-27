//! Qwen3.8 (Qwen3-Next style hybrid) model configuration.

use std::path::Path;

use apxinf_core::{Error, Result};

/// Configuration parsed from `config.json` (text_config + quantization_config).
#[derive(Debug, Clone)]
pub struct Qwen35Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
    /// Fraction of head_dim that receives rotary encoding (0.25 -> 64/256).
    pub partial_rotary_factor: f32,
    /// Every `full_attention_interval`-th layer is full attention (4).
    pub full_attention_interval: usize,
    /// Gate the attention output with sigmoid(gate) (bundled in q_proj).
    pub attn_output_gate: bool,
    // linear-attention (Gated DeltaNet) geometry.
    pub linear_key_head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_value_head_dim: usize,
    pub linear_num_value_heads: usize,
    pub linear_conv_kernel_dim: usize,
    /// SSM recurrence dtype ("float32").
    pub mamba_ssm_dtype: String,
    // quantization
    pub quant_group_size: usize,
    pub quant_num_bits: usize,
}

impl Qwen35Config {
    pub fn rotary_dim(&self) -> usize {
        (self.head_dim as f32 * self.partial_rotary_factor) as usize
    }

    pub fn linear_key_dim(&self) -> usize {
        self.linear_key_head_dim * self.linear_num_key_heads
    }

    pub fn linear_value_dim(&self) -> usize {
        self.linear_value_head_dim * self.linear_num_value_heads
    }

    pub fn is_full_attention(&self, layer_idx: usize) -> bool {
        layer_idx % self.full_attention_interval == self.full_attention_interval - 1
    }

    pub fn num_full_attention_layers(&self) -> usize {
        (self.n_layers + self.full_attention_interval - 1) / self.full_attention_interval
    }

    pub fn from_json_file(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path)
            .map_err(|e| Error::Other(format!("read {}: {e}", path.display())))?;
        Self::from_json(&raw)
    }

    pub fn from_json(raw: &str) -> Result<Self> {
        let root: serde_json::Value = serde_json::from_str(raw)
            .map_err(|e| Error::Other(format!("parse config.json: {e}")))?;
        let tc = root
            .get("text_config")
            .ok_or_else(|| Error::Other("config.json has no text_config".to_string()))?;

        let get = |key: &str| -> Result<usize> {
            tc.get(key)
                .and_then(serde_json::Value::as_u64)
                .map(|v| v as usize)
                .ok_or_else(|| Error::Other(format!("text_config.{key} missing")))
        };
        let get_f = |key: &str| -> Result<f32> {
            tc.get(key)
                .and_then(serde_json::Value::as_f64)
                .map(|v| v as f32)
                .ok_or_else(|| Error::Other(format!("text_config.{key} missing")))
        };

        let rope_theta = tc
            .get("rope_parameters")
            .and_then(|r| r.get("rope_theta"))
            .and_then(serde_json::Value::as_f64)
            .map(|v| v as f32)
            .unwrap_or(1_000_000.0);

        let partial_rotary_factor = tc
            .get("partial_rotary_factor")
            .and_then(serde_json::Value::as_f64)
            .map(|v| v as f32)
            .or_else(|| {
                tc.get("rope_parameters")
                    .and_then(|r| r.get("partial_rotary_factor"))
                    .and_then(serde_json::Value::as_f64)
                    .map(|v| v as f32)
            })
            .unwrap_or(1.0);

        // quantization_config group size / num bits
        let qc = root.get("quantization_config");
        let (quant_group_size, quant_num_bits) = match qc {
            Some(qc) => {
                let gs = qc
                    .get("config_groups")
                    .and_then(|g| g.get("group_0"))
                    .and_then(|g| g.get("weights"))
                    .and_then(|w| w.get("group_size"))
                    .and_then(serde_json::Value::as_u64)
                    .map(|v| v as usize)
                    .unwrap_or(32);
                let nb = qc
                    .get("config_groups")
                    .and_then(|g| g.get("group_0"))
                    .and_then(|g| g.get("weights"))
                    .and_then(|w| w.get("num_bits"))
                    .and_then(serde_json::Value::as_u64)
                    .map(|v| v as usize)
                    .unwrap_or(4);
                (gs, nb)
            }
            None => (32, 4),
        };

        let hidden_size = get("hidden_size")?;
        let n_heads = get("num_attention_heads")?;
        let head_dim = tc
            .get("head_dim")
            .and_then(serde_json::Value::as_u64)
            .map(|v| v as usize)
            .unwrap_or(hidden_size / n_heads);

        Ok(Self {
            hidden_size,
            intermediate_size: get("intermediate_size")?,
            n_layers: get("num_hidden_layers")?,
            n_heads,
            n_kv_heads: get("num_key_value_heads")?,
            head_dim,
            vocab_size: get("vocab_size")?,
            max_seq_len: get("max_position_embeddings")?.min(16640),
            rope_theta,
            rms_norm_eps: get_f("rms_norm_eps")?,
            partial_rotary_factor,
            full_attention_interval: get("full_attention_interval")?,
            attn_output_gate: tc
                .get("attn_output_gate")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(true),
            linear_key_head_dim: get("linear_key_head_dim")?,
            linear_num_key_heads: get("linear_num_key_heads")?,
            linear_value_head_dim: get("linear_value_head_dim")?,
            linear_num_value_heads: get("linear_num_value_heads")?,
            linear_conv_kernel_dim: get("linear_conv_kernel_dim")?,
            mamba_ssm_dtype: tc
                .get("mamba_ssm_dtype")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("float32")
                .to_string(),
            quant_group_size,
            quant_num_bits,
        })
    }
}
