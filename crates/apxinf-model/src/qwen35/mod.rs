//! Qwen3.8-27B (Qwen3-Next style hybrid attention) model implementation.

pub mod config;
pub mod general;
pub mod weights;

use std::path::Path;
use std::sync::Arc;

use apxinf_core::{Backend, Device, Error, Result};

use crate::auto::{LoadOptions, LoadedModel};

use config::Qwen35Config;
use general::GeneralQwen35;
use weights::Qwen35Weights;

/// Factory registered under the `qwen3_5` model name.
pub fn load_qwen35(
    path: &Path,
    device: Device,
    backend: Arc<dyn Backend>,
    _options: &LoadOptions,
) -> Result<LoadedModel> {
    if !matches!(device, Device::Cuda(_)) {
        return Err(Error::Other(
            "qwen3_5 requires a CUDA device (RTX 4090)".into(),
        ));
    }
    let model_dir = if path.is_dir() {
        path
    } else {
        path.parent().unwrap_or_else(|| Path::new("."))
    };
    let config = Qwen35Config::from_json_file(&model_dir.join("config.json"))?;
    let (tensors, _) = apxinf_loader::safetensors::load_native_path(path)
        .map_err(|e| Error::Other(format!("load {}: {e}", path.display())))?;
    let weights = Qwen35Weights::from_map(&config, tensors)?;
    let model = GeneralQwen35::new(config, weights, backend)?;
    Ok(LoadedModel::Text(Box::new(model)))
}
