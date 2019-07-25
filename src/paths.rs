
use std::path::PathBuf;

use super::model_info::ModelInfo;

pub struct Paths<'a> {
    model_info: &'a ModelInfo
}

impl<'a> Paths<'a> {
    pub fn from_model_info(model_info: &'a ModelInfo) -> Paths {
        Paths {
            model_info
        }
    }

    pub fn get_base_game_path(&self) -> PathBuf {
        PathBuf::from(format!(
            "./{}_runs",
            self.model_info.get_game_name()
        ))
    }

    pub fn get_base_path(&self) -> PathBuf {
        self.get_base_game_path().join(self.model_info.get_run_name())
    }

    pub fn get_models_path(&self) -> PathBuf {
        self.get_base_path().join("models")
    }

    pub fn get_exported_models_path(&self) -> PathBuf {
        self.get_base_path().join("exported_models")
    }

    pub fn get_games_path(&self) -> PathBuf {
        self.get_base_path().join("games")
    }
}