use anyhow::{anyhow, Result};
use log::info;
use rand::prelude::SliceRandom;
use serde::{Deserialize, Serialize};
use std::ffi::OsString;
use std::fs::{self, OpenOptions};
use std::fs::{DirEntry, File};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

pub struct DirIndex {
    path: PathBuf,
    files: Files,
    created: SystemTime,
    modified: SystemTime,
}

impl DirIndex {
    pub fn new(entry: &DirEntry) -> Result<Self> {
        let path = entry.path();
        let meta = entry.metadata()?;

        let mut inner_self = DirIndex {
            path,
            files: Files::Unexpanded(0),
            created: meta.created()?,
            modified: meta.modified()?,
        };

        inner_self.index()?;

        Ok(inner_self)
    }

    pub fn created(&self) -> SystemTime {
        self.created
    }

    pub fn index(&mut self) -> Result<()> {
        // If the cache is up to date, then just set the number of games according to the cache.
        if let Ok(cache) = self.load_cache_current() {
            if matches!(self.files, Files::Unexpanded(_)) {
                self.files = Files::Unexpanded(cache.games);
            }
            info!(
                "Cache is up to date for {:?}",
                self.path.components().next_back()
            );
            return Ok(());
        }

        self.write_cache()?;

        Ok(())
    }

    pub fn num_games(&self) -> usize {
        self.files.num_games()
    }

    pub fn iter(&mut self) -> impl DoubleEndedIterator<Item = PathBuf> + '_ {
        self.files.iter(&self.path)
    }

    pub fn expand(&mut self) -> Result<()> {
        self.files.expanded(&self.path)?;
        Ok(())
    }

    pub fn sample(&self) -> PathBuf {
        let mut rng = rand::thread_rng();
        let sample = self
            .files
            .try_expanded()
            .expect("Directory should have already been expanded")
            .choose(&mut rng)
            .expect("No samples found in dir");

        self.path.join(&sample.0)
    }

    fn load_cache_current(&self) -> Result<DirCache> {
        let file = File::open(self.cache_path())?;
        if file.metadata()?.modified()? >= self.modified - Duration::from_secs(1) {
            return Ok(serde_json::from_reader(file)?);
        }

        Err(anyhow!("Cache does not exist or is out of date"))
    }

    fn load_cache(&self) -> Result<DirCache> {
        let file = File::open(self.cache_path())?;

        Ok(serde_json::from_reader(file)?)
    }

    fn cache_path(&self) -> PathBuf {
        self.path.join("cache.json")
    }

    fn write_cache(&mut self) -> Result<()> {
        let cache_path = self.cache_path();

        let entries = self.files.expanded(&self.path)?;
        let num_files = entries.len();

        // Attempt to write a cache file. It is OK if this fails.
        let _res: Result<usize> = (|| {
            let mut cache_lock_path = cache_path.clone();
            cache_lock_path.set_extension("lock");

            OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&cache_lock_path)?;

            let num_cache_entries = self.load_cache().map(|d| d.games).unwrap_or(0);

            let cache_file = File::create(cache_path)?;

            if num_cache_entries > num_files {
                info!("Cache file is mtime is out of date but has more vals than in directory. Updating cache entry {:?}", self.cache_path());
            }

            serde_json::to_writer_pretty(
                cache_file,
                &DirCache {
                    games: num_files.max(num_cache_entries),
                },
            )?;

            fs::remove_file(cache_lock_path)?;

            Ok(0)
        })();

        Ok(())
    }
}

enum Files {
    Unexpanded(usize),
    Expanded(Vec<(OsString, SystemTime)>),
}

impl Files {
    fn num_games(&self) -> usize {
        match self {
            Self::Unexpanded(count) => *count,
            Self::Expanded(files) => files.len(),
        }
    }

    fn try_expanded(&self) -> Option<&Vec<(OsString, SystemTime)>> {
        match self {
            Files::Expanded(entries) => Some(entries),
            Files::Unexpanded(_) => None,
        }
    }

    fn expanded(&mut self, path: &Path) -> Result<&mut Vec<(OsString, SystemTime)>> {
        if let Files::Expanded(entries) = self {
            return Ok(entries);
        }

        info!("Expanding {:?}", path.components().next_back());

        let mut entries = path
            .read_dir()?
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_ok_and(|f| f.is_file()))
            .filter(|e| {
                e.path()
                    .extension()
                    .map(|ext| ext.to_string_lossy())
                    .is_some_and(|ext| ext == "gz")
            })
            .filter_map(|e| {
                let meta_data = e.metadata().ok()?;
                Some((e.file_name(), meta_data.created().ok()?))
            })
            .collect::<Vec<_>>();

        entries.sort_by_key(|(_, created)| *created);

        *self = Files::Expanded(entries);

        self.expanded(path)
    }

    pub fn iter<'a>(&'a mut self, path: &'a Path) -> impl DoubleEndedIterator<Item = PathBuf> + 'a {
        self.expanded(path)
            .expect("Failed to expand dir")
            .iter()
            .map(|(file_name, _)| path.join(file_name))
    }
}

#[derive(Serialize, Deserialize)]
struct DirCache {
    games: usize,
}

#[cfg(test)]
mod test {
    #[test]
    fn load_index_test() {}
}
