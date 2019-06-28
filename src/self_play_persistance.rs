use std::path::Path;
use serde::Serialize;
use std::fs;
use std::io::Write;
use std::fs::{File,OpenOptions};

use super::self_play::{SelfPlayMetrics};

pub struct SelfPlayPersistance
{
    file: File
}

impl SelfPlayPersistance
{
    pub fn new(run_directory: &Path, model_name: &str) -> Result<Self, &'static str> {
        // @TODO: Add run name to file here.
        let games_dir = run_directory.join("games");
        let file_path = games_dir.join(format!("{}.json", model_name));

        fs::create_dir_all(games_dir).map_err(|_| "Couldn't create games dir")?;

        let file = OpenOptions::new()
            .append(true)
            .create(true)
            .open(file_path)
            .map_err(|_| "Couldn't open or create the self_play file")?;

        Ok(Self {
            file
        })
    }

    pub fn write<A: Serialize>(&mut self, self_play_metrics: SelfPlayMetrics<A>) -> Result<(), &'static str> {
        let serialized = serde_json::to_string(&self_play_metrics)
            .map_err(|_| "Failed to serialize results")?;

        writeln!(self.file, "{}", serialized).map_err(|_| "Failed to write to self_play file")?;

        Ok(())
    }
}
