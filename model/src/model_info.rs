use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct ModelInfo {
    game_name: String,
    run_name: String,
    model_num: usize,
}

impl ModelInfo {
    pub fn new(game_name: String, run_name: String, model_num: usize) -> ModelInfo {
        ModelInfo {
            game_name,
            run_name,
            model_num,
        }
    }

    pub fn from_model_name(model_name: &str) -> ModelInfo {
        let parts: Vec<_> = model_name.split('_').collect();

        ModelInfo {
            game_name: parts[0].to_string(),
            run_name: parts[1].to_string(),
            model_num: parts[2].parse().unwrap(),
        }
    }

    pub fn is_model_name(model_name: &str) -> bool {
        let parts: Vec<_> = model_name.split('_').collect();
        parts.len() == 3
            && parts[2].split('.').collect::<Vec<_>>()[0]
                .parse::<usize>()
                .is_ok()
    }

    pub fn get_game_name(&self) -> &str {
        &self.game_name
    }

    pub fn get_run_name(&self) -> &str {
        &self.run_name
    }

    pub fn get_model_num(&self) -> usize {
        self.model_num
    }

    pub fn get_model_name(&self) -> String {
        format!(
            "{}_{}_{:0>5}",
            self.game_name, self.run_name, self.model_num
        )
    }

    pub fn get_next_model_info(&self) -> ModelInfo {
        ModelInfo {
            game_name: self.game_name.to_owned(),
            run_name: self.run_name.to_owned(),
            model_num: self.model_num + 1,
        }
    }
}
