use std::fs;

use super::model::{Model};
use super::paths::Paths;

use model::model_info::ModelInfo;

pub struct ModelFactory {}

impl ModelFactory {
    pub fn new() -> Self {
        Self {}
    }
}

impl model::model::ModelFactory for ModelFactory {
    type M = Model;

    fn create(&self, name: &str, num_filters: usize, num_blocks: usize) -> Model {
        // @TODO: Replace with code to create the model.
        get_latest(name).expect("Failed to get latest model")        
    }

    fn get_latest(&self, name: &str) -> Model {
        get_latest(name).expect("Failed to get latest model")
    }
}

fn get_latest(name: &str) -> std::io::Result<Model> {
    let model_info = ModelInfo::from_model_name(name);
    let paths = Paths::from_model_info(&model_info);

    let latest_run_num = fs::read_dir(paths.get_models_path())?
        .map(|e| e.expect("Could not read model file").path())
        .filter(|p| p.is_file())
        .filter_map(|p| {
            p.file_name().and_then(|p| p.to_str()).map(|s| s.to_owned())
        })
        .map(|n| {
            let model_name_excluding_file_ext = &n[0..(n.len() - 3)];
            ModelInfo::from_model_name(model_name_excluding_file_ext).get_run_num()
        })
        .max()
        .expect("No models found");

    let latest_model_name = ModelInfo::new(
            model_info.get_game_name().to_owned(),
            model_info.get_run_name().to_owned(),
            latest_run_num
        )
        .get_model_name();

    println!("Getting latest model: {}", latest_model_name);

    Ok(Model::new(latest_model_name))
}