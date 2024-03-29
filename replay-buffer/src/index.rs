use crate::dir_index::*;
use anyhow::Result;
use log::{error, info, warn};
use rand::prelude::*;
use std::{ops::Range, path::PathBuf, time::Instant};

pub struct Index {
    indexes: Vec<DirIndex>,
    games_dir: PathBuf,
}

impl Index {
    pub fn new(games_dir: PathBuf) -> Result<Self> {
        let mut _self = Self {
            indexes: vec![],
            games_dir,
        };

        _self.re_index()?;

        Ok(_self)
    }

    pub fn sampler(&mut self, range: Range<usize>) -> Result<Sampler<'_>> {
        Sampler::new(range, &mut self.indexes)
    }

    pub fn games(&mut self) -> usize {
        self.indexes.iter().map(|indx| indx.num_games()).sum()
    }

    pub fn re_index(&mut self) -> Result<()> {
        let mut indexes = self
            .games_dir
            .read_dir()?
            .flatten()
            .filter(|e| e.file_type().is_ok_and(|e| e.is_dir()))
            .filter_map(|e| {
                let index = DirIndex::new(&e);
                if let Err(err) = &index {
                    error!("Failed to index directory: {:?} {:?}", e.path(), err);
                }

                index.ok()
            })
            .collect::<Vec<_>>();

        indexes.sort_by_key(|e| e.created());

        self.indexes = indexes;

        Ok(())
    }

    pub fn iter(&mut self) -> impl DoubleEndedIterator<Item = PathBuf> + '_ {
        self.indexes.iter_mut().flat_map(|indx| indx.iter())
    }
}

pub struct Sampler<'a> {
    range: Range<usize>,
    indexes: &'a Vec<DirIndex>,
}

impl<'a> Sampler<'a> {
    pub fn new(range: Range<usize>, indexes: &'a mut Vec<DirIndex>) -> Result<Sampler> {
        let start = Instant::now();
        let mut index_start = 0;
        let mut index_end = 0;
        for index in indexes.iter_mut() {
            index_end += index.num_games();

            if range.start < index_end {
                index.expand()?;
            }

            if range.end < index_start {
                break;
            }

            index_start += index.num_games();
        }

        if start.elapsed().as_millis() > 1 {
            info!("Expanded in {} milliseconds", start.elapsed().as_millis());
        }

        Ok(Self { range, indexes })
    }

    pub fn sample(&self) -> PathBuf {
        let mut rng = rand::thread_rng();
        let sample_idx = rng.gen_range(self.range.clone());

        let mut index_end = 0;
        for index in self.indexes.iter() {
            index_end += index.num_games();
            if sample_idx < index_end {
                return index.sample();
            }
        }

        warn!("Sample Index: {} is beyond available number of indexes: {}. There are not as many games as indicated by the index.", sample_idx, index_end);

        self.indexes
            .iter()
            .last()
            .expect("Expected at least one index in indexes to exist")
            .sample()
    }
}
