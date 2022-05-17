use anyhow::{Context as AnyhowContext, Result};
use std::fs::File;
use std::io::BufReader;
use std::io::Write;
use std::path::PathBuf;

use super::super::ModelInfo;
use super::{get_model_dir, TensorflowModelOptions};

pub fn get_model_options_path(model_info: &ModelInfo) -> PathBuf {
    get_model_dir(model_info).join("model-options.json")
}

pub fn get_options(model_info: &ModelInfo) -> Result<TensorflowModelOptions> {
    let file_path = get_model_options_path(model_info);
    let file_path_lossy = format!("{}", file_path.to_string_lossy());
    let file = File::open(file_path).context(file_path_lossy)?;
    let reader = BufReader::new(file);
    let options = serde_json::from_reader(reader)?;
    Ok(options)
}

pub fn write_options(model_info: &ModelInfo, options: &TensorflowModelOptions) -> Result<()> {
    let serialized_options = serde_json::to_string(options)?;

    let file_path = get_model_options_path(model_info);
    let file_path_lossy = format!("{}", file_path.to_string_lossy());
    let mut file = File::create(file_path).context(file_path_lossy)?;
    writeln!(file, "{}", serialized_options)?;

    Ok(())
}
