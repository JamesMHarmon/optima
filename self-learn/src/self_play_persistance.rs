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
        Self::read_metrics_from_file(&file_path)
    }

    pub fn read_all_reverse_iter<A: DeserializeOwned>(&self) -> Result<SelfPlayMetricsIterator<A>, &'static str> {
        let file_paths = Self::get_game_files_in_dir(&self.games_dir)?;

        Ok(SelfPlayMetricsIterator::new(file_paths))
    }

    fn read_metrics_from_file<A: DeserializeOwned>(file_path: &Path) -> Result<Vec<SelfPlayMetrics<A>>, &'static str> {
        let file = File::open(file_path);

        match file {
            Err(_) => Ok(Vec::new()),
            Ok(file) => {
                let buf = BufReader::new(file);
                let games: Vec<SelfPlayMetrics<A>> = buf.lines()
                    .enumerate()
                    .filter_map(|(i,l)| l.ok().map(|l| (i,l)))
                    .map(|(i,l)| serde_json::from_str(&l).map_err(|_| format!("Failed to deserialize {:?} at line: {}", file_path, i)).unwrap())
                    .collect();

                Ok(games)
            }
        }
    }

    fn get_file_path(games_dir: &Path, name: &str) -> PathBuf {
        games_dir.join(format!("{}.json", name))
    }

    fn get_game_files_in_dir(games_dir: &Path) -> Result<Vec<PathBuf>, &'static str> {
        let file_paths: Vec<PathBuf> = fs::read_dir(games_dir).map_err(|_| "Error reading games_dir")?
            .filter_map(|e| e.map(|e| e.path()).ok())
            .filter(|p| p.is_file())
            .collect();

        Ok(file_paths)
    }
}

pub struct SelfPlayMetricsIterator<A>
where
    A: DeserializeOwned
{
    file_paths: Vec<PathBuf>,
    metrics: Vec<SelfPlayMetrics<A>>
}

impl<A> SelfPlayMetricsIterator<A>
where
    A: DeserializeOwned
{
    pub fn new(file_paths: Vec<PathBuf>) -> Self {
        Self { file_paths, metrics: Vec::new() }
    }
}

impl<A> Iterator for SelfPlayMetricsIterator<A>
where
    A: DeserializeOwned
{
    type Item = SelfPlayMetrics<A>;

    fn next(&mut self) -> Option<SelfPlayMetrics<A>> {
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
