use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct ModelInfo {
    game_name: String,
    run_name: String,
    model_name: String,
    model_num: usize,
}

impl ModelInfo {
    pub fn new(
        game_name: String,
        run_name: String,
        model_name: String,
        model_num: usize,
    ) -> ModelInfo {
        ModelInfo {
            game_name,
            run_name,
            model_name,
            model_num,
        }
    }

    pub fn game_name(&self) -> &str {
        &self.game_name
    }

    pub fn run_name(&self) -> &str {
        &self.run_name
    }

    pub fn model_num(&self) -> usize {
        self.model_num
    }

    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    pub fn model_name_w_num(&self) -> String {
        format!("{}_{}", self.model_name(), self.model_num())
    }
}
