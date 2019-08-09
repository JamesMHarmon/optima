use std::path::{Path,PathBuf};
use serde::{Serialize};
use std::fs;
use std::io::Write;
use std::fs::{File,OpenOptions};

use model::model_info::ModelInfo;
use super::self_evaluate::{GameResult,MatchResult};

pub struct SelfEvaluatePersistance
{
    game_file: File,
    match_file: File
}

impl SelfEvaluatePersistance
{
    pub fn new(run_directory: &Path, model_info_1: &ModelInfo, model_info_2: &ModelInfo) -> Result<Self, &'static str> {
        let evaluations_dir = run_directory.join("evaluations");

        let game_file_path = get_game_file_path(
            &evaluations_dir,
            model_info_1,
            model_info_2
        );

        let match_file_path = get_match_file_path(&evaluations_dir);

        fs::create_dir_all(&evaluations_dir).map_err(|_| "Couldn't create games dir")?;

        let game_file = OpenOptions::new()
            .append(true)
            .create(true)
            .open(game_file_path)
            .map_err(|_| "Couldn't open or create the game_file file")?;
        
        let match_file = OpenOptions::new()
            .append(true)
            .create(true)
            .open(match_file_path)
            .map_err(|_| "Couldn't open or create the match_file file")?;

        Ok(Self {
            game_file,
            match_file
        })
    }

    pub fn write_game<A: Serialize>(&mut self, game_result: &GameResult<A>) -> Result<(), &'static str> {
        let serialized = serde_json::to_string(game_result)
            .map_err(|_| "Failed to serialize results")?;

        writeln!(self.game_file, "{}", serialized).map_err(|_| "Failed to write to game_result file")?;

        Ok(())
    }

    pub fn write_match(&mut self, match_result: &MatchResult) -> Result<(), &'static str> {
        let serialized = serde_json::to_string(match_result)
            .map_err(|_| "Failed to serialize results")?;

        writeln!(self.match_file, "{}", serialized).map_err(|_| "Failed to write to match_result file")?;

        Ok(())
    }
}

fn get_game_file_path(evaluations_dir: &Path, model_info_1: &ModelInfo, model_info_2: &ModelInfo) -> PathBuf {
    evaluations_dir.join(format!(
        "{model_1_name}_vs_{model_2_name}.json",
        model_1_name = model_info_1.get_model_name(),
        model_2_name = model_info_2.get_model_name(),
    ))
}

fn get_match_file_path(evaluations_dir: &Path) -> PathBuf {
    evaluations_dir.join("match_results.json")
}
