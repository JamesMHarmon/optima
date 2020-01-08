use std::path::{Path,PathBuf};
use serde::{Serialize};
use std::fs;
use std::io::Write;
use std::fs::{File,OpenOptions};
use failure::Error;

use super::tuner::{GameResult,PlayerScore};

pub struct TunerPersistance
{
    game_file: File,
    match_file: File
}

impl TunerPersistance
{
    pub fn new(run_directory: &Path, name: &str) -> Result<Self, Error> {
        let tuner_dir = run_directory.join("tuner");

        let game_file_path = get_game_file_path(&tuner_dir, name);
        let match_file_path = get_match_file_path(&tuner_dir, name);

        fs::create_dir_all(&tuner_dir)?;
        
        let game_file = OpenOptions::new()
            .create(true)
            .write(true)
            .open(game_file_path).unwrap();
        
        let match_file = OpenOptions::new()
            .create(true)
            .write(true)
            .open(match_file_path)?;

        Ok(Self {
            game_file,
            match_file
        })
    }

    pub fn write_game<A: Serialize>(&mut self, game_result: &GameResult<A>) -> Result<(), Error> {
        let serialized = serde_json::to_string(game_result)?;

        writeln!(self.game_file, "{}", serialized)?;

        Ok(())
    }

    pub fn write_player_scores(&mut self, player_scores: &[PlayerScore]) -> Result<(), Error> {
        let mut player_scores = player_scores.iter().collect::<Vec<_>>();
        player_scores.sort_by(|PlayerScore { score: s1, .. }, PlayerScore { score: s2, .. }| s1.partial_cmp(s2).unwrap());

        let serialized = serde_json::to_string_pretty(&player_scores)?;

        writeln!(self.match_file, "{}", serialized)?;

        Ok(())
    }
}

fn get_game_file_path(tuner_dir: &Path, name: &str) -> PathBuf {
    tuner_dir.join(format!(
        "{}_games.json", name
    ))
}

fn get_match_file_path(tuner_dir: &Path, name: &str) -> PathBuf {
    tuner_dir.join(format!(
        "{}_results.json", name
    ))
}
