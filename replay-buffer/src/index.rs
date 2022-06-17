use std::{path::{PathBuf, Path}, fs::DirEntry, time::SystemTime};
use anyhow::Result;
use rand::prelude::*;

pub struct Index {
    files: Vec<(DirEntry, SystemTime)>,
    games_dir: PathBuf
}

impl Index {
    pub fn new(games_dir: PathBuf) -> Result<Self> {
        let mut files = get_game_files(games_dir.as_ref())?;
    
        files.sort_by(|(_, a), (_, b)| b.cmp(a));

        println!("First: {:?}, Last: {:?}", files.first(), files.last());

        Ok(Self {
            files,
            games_dir
        })
    }

    pub fn sample(&self, start_idx: usize, end_idx: usize) -> PathBuf {
        let mut rng = rand::thread_rng();
        let sample_idx = rng.gen_range(start_idx..end_idx);

        self.files[sample_idx].0.path()
    }

    pub fn games(&self) -> usize {
        self.files.len()
    }
}

fn get_game_files(games_dir: &Path) -> Result<Vec<(DirEntry, SystemTime)>> {
    let mut files = vec![];

    for entry in games_dir.read_dir()?.flatten() {
        if entry.file_type()?.is_dir() {
            let games_in_dir = get_game_files(&entry.path())?;
            files.extend(games_in_dir);
        } else {
            let created = entry.metadata()?.created()?;
            files.push((entry, created));
        }
    }

    Ok(files)
}
