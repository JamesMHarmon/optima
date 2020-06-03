use anyhow::Result;
use serde::de::DeserializeOwned;
use std::path::{Path,PathBuf};
use serde::{Serialize};
use std::fs;
use std::io::{BufReader,BufRead,Write};
use std::fs::{File,OpenOptions};
use model::model_info::ModelInfo;
use log::info;

use super::self_play::{SelfPlayMetrics};

pub struct SelfPlayPersistance
{
    file: File,
    games_dir: PathBuf,
    model_name: String
}

impl SelfPlayPersistance
{
    pub fn new(run_directory: &Path, model_name: String) -> Result<Self> {
        let games_dir = run_directory.join("games");
        let file_path = Self::get_file_path(&games_dir, &model_name);

        fs::create_dir_all(&games_dir)?;

        let file = OpenOptions::new()
            .append(true)
            .create(true)
            .open(file_path)?;

        Ok(Self {
            games_dir,
            file,
            model_name
        })
    }

    pub fn write<A: Serialize, V: Serialize>(&mut self, self_play_metrics: &SelfPlayMetrics<A,V>) -> Result<()> {
        let serialized = serde_json::to_string(self_play_metrics)?;

        writeln!(self.file, "{}", serialized)?;

        Ok(())
    }

    pub fn read<A: DeserializeOwned, V: DeserializeOwned>(&self) -> Result<Box<dyn Iterator<Item=SelfPlayMetrics<A,V>>>> {
        let file_path = Self::get_file_path(&self.games_dir, &self.model_name);
        Self::read_metrics_from_file(&file_path)
    }

    pub fn read_all_reverse_iter<A: DeserializeOwned, V: DeserializeOwned>(&self, model_num: usize) -> Result<SelfPlayMetricsIterator<A,V>> {
        let file_paths = Self::get_game_files_in_dir(&self.games_dir)?;

        let mut file_paths = file_paths.into_iter().filter(|p| get_model_num(p) <= model_num).collect::<Vec<_>>();

        file_paths.sort();

        Ok(SelfPlayMetricsIterator::new(file_paths))
    }

    fn read_metrics_from_file<A: DeserializeOwned, V: DeserializeOwned>(file_path: &Path) -> Result<Box<dyn Iterator<Item=SelfPlayMetrics<A,V>>>> {
        info!("Reading File: {:?}", file_path);
        let file = File::open(file_path);
        let empty_iter = (0..0).take(0).map(|_| serde_json::from_str(&"").unwrap());

        match file {
            Err(_) => Ok(Box::new(empty_iter)),
            Ok(file) => {
                let buf = BufReader::new(file);
                let games = buf.lines()
                    .filter_map(|l| l.ok())
                    .map(|l| serde_json::from_str(&l).unwrap());

                Ok(Box::new(games))
            }
        }
    }

    fn get_file_path(games_dir: &Path, name: &str) -> PathBuf {
        games_dir.join(format!("{}.json", name))
    }

    fn get_game_files_in_dir(games_dir: &Path) -> Result<Vec<PathBuf>> {
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

fn get_model_num(path: &PathBuf) -> usize {
    let file_stem = path.file_stem().unwrap().to_str().unwrap();
    ModelInfo::from_model_name(file_stem).get_model_num()
}

pub struct SelfPlayMetricsIterator<A,V>
where
    A: DeserializeOwned,
    V: DeserializeOwned
{
    file_paths: Vec<PathBuf>,
    metrics: Box<dyn Iterator<Item=SelfPlayMetrics<A,V>>>
}

impl<A,V> SelfPlayMetricsIterator<A,V>
where
    A: DeserializeOwned,
    V: DeserializeOwned
{
    pub fn new(file_paths: Vec<PathBuf>) -> Self {
        let empty_iter = (0..0).take(0).map(|_| serde_json::from_str(&"").unwrap());
        Self { file_paths, metrics: Box::new(empty_iter) }
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
            let metric = self.metrics.next();

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
