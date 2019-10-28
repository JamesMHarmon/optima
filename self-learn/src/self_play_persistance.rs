use failure::Error;
use serde::de::DeserializeOwned;
use std::path::{Path,PathBuf};
use serde::{Serialize};
use std::fs;
use std::io::{BufReader,BufRead,Write};
use std::fs::{File,OpenOptions};
use model::model_info::ModelInfo;

use super::self_play::{SelfPlayMetrics};

pub struct SelfPlayPersistance
{
    file: File,
    games_dir: PathBuf,
    name: String
}

impl SelfPlayPersistance
{
    pub fn new(run_directory: &Path, name: String) -> Result<Self, Error> {
        let games_dir = run_directory.join("games");
        let file_path = Self::get_file_path(&games_dir, &name);

        fs::create_dir_all(&games_dir)?;

        let file = OpenOptions::new()
            .append(true)
            .create(true)
            .open(file_path)?;

        Ok(Self {
            games_dir,
            file,
            name
        })
    }

    pub fn write<A: Serialize, V: Serialize>(&mut self, self_play_metrics: &SelfPlayMetrics<A,V>) -> Result<(), Error> {
        let serialized = serde_json::to_string(self_play_metrics)?;

        writeln!(self.file, "{}", serialized)?;

        Ok(())
    }

    pub fn read<A: DeserializeOwned, V: DeserializeOwned>(&self) -> Result<Vec<SelfPlayMetrics<A,V>>, Error> {
        let file_path = Self::get_file_path(&self.games_dir, &self.name);
        Self::read_metrics_from_file(&file_path)
    }

    pub fn read_all_reverse_iter<A: DeserializeOwned, V: DeserializeOwned>(&self) -> Result<SelfPlayMetricsIterator<A,V>, Error> {
        let mut file_paths = Self::get_game_files_in_dir(&self.games_dir)?;

        file_paths.sort();

        Ok(SelfPlayMetricsIterator::new(file_paths))
    }

    fn read_metrics_from_file<A: DeserializeOwned, V: DeserializeOwned>(file_path: &Path) -> Result<Vec<SelfPlayMetrics<A,V>>, Error> {
        let file = File::open(file_path);

        match file {
            Err(_) => Ok(Vec::new()),
            Ok(file) => {
                let buf = BufReader::new(file);
                let games: Vec<SelfPlayMetrics<A,V>> = buf.lines()
                    .enumerate()
                    .filter_map(|(i,l)| l.ok().map(|l| (i,l)))
                    .map(|(_,l)| serde_json::from_str(&l).unwrap())
                    .collect();

                Ok(games)
            }
        }
    }

    fn get_file_path(games_dir: &Path, name: &str) -> PathBuf {
        games_dir.join(format!("{}.json", name))
    }

    fn get_game_files_in_dir(games_dir: &Path) -> Result<Vec<PathBuf>, Error> {
        let file_paths: Vec<PathBuf> = fs::read_dir(games_dir)?
            .filter_map(|e| e.map(|e| e.path()).ok())
            .filter(|p| p.is_file() && file_path_is_valid_game_file(p))
            .collect();

        Ok(file_paths)
    }
}

fn file_path_is_valid_game_file(path: &PathBuf) -> bool {
    if let Some(game_name) = path.file_stem() {
        if let Some(game_name) = game_name.to_str() {
            return ModelInfo::is_model_name(game_name)
        }
    }

    false
}

pub struct SelfPlayMetricsIterator<A,V>
where
    A: DeserializeOwned,
    V: DeserializeOwned
{
    file_paths: Vec<PathBuf>,
    metrics: Vec<SelfPlayMetrics<A,V>>
}

impl<A,V> SelfPlayMetricsIterator<A,V>
where
    A: DeserializeOwned,
    V: DeserializeOwned
{
    pub fn new(file_paths: Vec<PathBuf>) -> Self {
        Self { file_paths, metrics: Vec::new() }
    }
}

impl<A,V> Iterator for SelfPlayMetricsIterator<A,V>
where
    A: DeserializeOwned,
    V: DeserializeOwned
{
    type Item = SelfPlayMetrics<A,V>;

    fn next(&mut self) -> Option<SelfPlayMetrics<A,V>> {
        loop {
            let metric = self.metrics.pop();

            if metric.is_some() {
                return metric;
            }

            let file_path = self.file_paths.pop();

            if file_path.is_none() {
                return None;
            }

            let metrics = SelfPlayPersistance::read_metrics_from_file(&file_path.unwrap()).unwrap();

            self.metrics = metrics;
        }
    }
}
