use crate::dir_index::*;
use anyhow::Result;
use log::error;
use rand::prelude::*;
use std::{ops::Range, path::PathBuf};

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
        let mut index_end = 0;
        for index in indexes.iter_mut() {
            index_end += index.num_games();

            if range.start < index_end {
                index.expand()?;
            }

            if range.end <= index_end {
                return Ok(Self { range, indexes });
            }
        }

        Ok(Self { range, indexes })
    }

    pub fn sample(&self) -> PathBuf {
        let mut rng = rand::thread_rng();
        let sample_idx = rng.gen_range(self.range.clone());

        let mut counts = 0;
        for index in self.indexes.iter() {
            counts += index.num_games();
            if sample_idx < counts {
                index.sample();
            }
        }

        panic!("Should not be reachable")
    }
}
