pub struct ModelInfo {
    game_name: String,
    run_name: String,
    run_num: usize
}

impl ModelInfo {
    pub fn new(game_name: String, run_name: String, run_num: usize) -> ModelInfo {
        ModelInfo {
            game_name,
            run_name,
            run_num
        }
    }

    pub fn from_model_name(model_name: &str) -> ModelInfo {
        let parts: Vec<_> = model_name.split("_").collect();

        ModelInfo {
            game_name: parts[0].to_string(),
            run_name: parts[1].to_string(),
            run_num: parts[2].parse().unwrap()
        }
    }

    pub fn get_game_name(&self) -> &str {
        &self.game_name
    }

    pub fn get_run_name(&self) -> &str {
        &self.run_name
    }

    pub fn get_run_num(&self) -> usize {
        self.run_num
    }

    pub fn get_model_name(&self) -> String {
        format!(
            "{}_{}_{:0>5}",
            self.game_name,
            self.run_name,
            self.run_num
        )
    }
}



