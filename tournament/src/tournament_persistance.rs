use anyhow::Result;
use serde::Serialize;
use std::fs;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use super::tournament::GameResult;
use model::model_info::ModelInfo;

pub struct TournamentPersistance {
    game_file: File,
    match_file: File,
}

impl TournamentPersistance {
    pub fn new(run_directory: &Path, model_infos: &[ModelInfo]) -> Result<Self> {
        let tournament_dir = run_directory.join("tournament");

        let game_file_path = get_game_file_path(&tournament_dir, model_infos);
        let match_file_path = get_match_file_path(&tournament_dir, model_infos);

        fs::create_dir_all(&tournament_dir)?;

        let game_file = OpenOptions::new()
            .append(true)
            .create(true)
            .open(game_file_path)?;

        let match_file = OpenOptions::new()
            .append(true)
            .create(true)
            .open(match_file_path)?;

        Ok(Self {
            game_file,
            match_file,
        })
    }

    pub fn write_game<A: Serialize>(&mut self, game_result: &GameResult<A>) -> Result<()> {
        let serialized = serde_json::to_string(game_result)?;

        writeln!(self.game_file, "{}", serialized)?;

        Ok(())
    }

    pub fn write_model_scores(&mut self, model_scores: &[(ModelInfo, f32)]) -> Result<()> {
        let mut model_scores = model_scores.to_owned();
        model_scores.sort_by(|(_, s1), (_, s2)| s1.partial_cmp(s2).unwrap());

        let serialized = serde_json::to_string_pretty(&model_scores)?;

        writeln!(self.match_file, "{}", serialized)?;

        Ok(())
    }
}

fn get_game_file_path(tournament_dir: &Path, model_infos: &[ModelInfo]) -> PathBuf {
    let min = model_infos.iter().map(|m| m.get_model_num()).min().unwrap();
    let max = model_infos.iter().map(|m| m.get_model_num()).max().unwrap();

    tournament_dir.join(format!("{}-{}_games.json", min, max))
}

fn get_match_file_path(tournament_dir: &Path, model_infos: &[ModelInfo]) -> PathBuf {
    let min = model_infos.iter().map(|m| m.get_model_num()).min().unwrap();
    let max = model_infos.iter().map(|m| m.get_model_num()).max().unwrap();

    tournament_dir.join(format!("{}-{}_results.json", min, max))
}
