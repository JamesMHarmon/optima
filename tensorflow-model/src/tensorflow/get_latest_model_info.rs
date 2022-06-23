use anyhow::{anyhow, Result};
use log::info;
use std::fs;

use super::paths::Paths;
use ::model::ModelInfo;

pub fn get_latest_model_info(model_info: &ModelInfo) -> Result<ModelInfo> {
    let paths = Paths::from_model_info(model_info);

    let latest_model_num = fs::read_dir(paths.get_models_path())?
        .map(|e| e.expect("Could not read model file").path())
        .filter(|p| p.is_file())
        .filter_map(|p| p.file_name().and_then(|p| p.to_str()).map(|s| s.to_owned()))
        .filter(|n| ModelInfo::is_model_name(n))
        .map(|n| {
            let model_name_excluding_file_ext = &n[0..(n.len() - 3)];
            ModelInfo::from_model_name(model_name_excluding_file_ext).get_model_num()
        })
        .max()
        .ok_or_else(|| anyhow!("No models found"))?;

    let latest_model_info = ModelInfo::new(
        model_info.get_game_name().to_owned(),
        model_info.get_run_name().to_owned(),
        latest_model_num,
    );

    info!(
        "Getting latest model: {}",
        latest_model_info.get_model_name()
    );

    Ok(latest_model_info)
}
