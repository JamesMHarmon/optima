use serde::de::DeserializeOwned;
use std::path::{Path,PathBuf};
use serde::{Serialize};
use std::fs;
use std::io::{BufReader,BufRead,Write};
use std::fs::{File,OpenOptions};

use super::self_play::{SelfPlayMetrics};

pub struct SelfPlayPersistance
{
    file: File,
    games_dir: PathBuf,
    name: String
}

impl SelfPlayPersistance
{
    pub fn new(run_directory: &Path, name: String) -> Result<Self, &'static str> {
        let games_dir = run_directory.join("games");
        let file_path = Self::get_file_path(&games_dir, &name);

        fs::create_dir_all(&games_dir).map_err(|_| "Couldn't create games dir")?;

        let file = OpenOptions::new()
            .append(true)
            .create(true)
            .open(file_path)
            .map_err(|_| "Couldn't open or create the self_play file")?;

        Ok(Self {
            games_dir,
            file,
            name
        })
    }

    pub fn write<A: Serialize>(&mut self, self_play_metrics: &SelfPlayMetrics<A>) -> Result<(), &'static str> {
        let serialized = serde_json::to_string(self_play_metrics)
            .map_err(|_| "Failed to serialize results")?;

        writeln!(self.file, "{}", serialized).map_err(|_| "Failed to write to self_play file")?;

        Ok(())
    }

    pub fn read<A: DeserializeOwned>(&self) -> Result<Vec<SelfPlayMetrics<A>>, &'static str> {
        let file_path = Self::get_file_path(&self.games_dir, &self.name);
        let file = File::open(file_path);

        match file {
            Err(_) => Ok(Vec::new()),
            Ok(file) => {
                let buf = BufReader::new(file);
                let games: Vec<SelfPlayMetrics<A>> = buf.lines()
                    .filter_map(|l| l.ok())
                    .filter_map(|l| serde_json::from_str(&l).ok())
                    .collect();

                Ok(games)
            }
        }
    }

    fn get_file_path(games_dir: &Path, name: &str) -> PathBuf {
        games_dir.join(format!("{}.json", name))
    }
}
