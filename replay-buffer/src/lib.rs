#![feature(try_blocks)]
#![feature(let_chains)]
#![feature(assert_matches)]
#![feature(test)]
#![feature(bench_black_box)]
#![feature(is_some_with)]

use anyhow::{Context, Result};
use engine::GameState;
use env_logger::Env;
use flate2::read::GzDecoder;
use log::warn;
use model::PositionMetrics;
use pyo3::types::{IntoPyDict, PyDict};
use rand::Rng;
use rayon::prelude::*;
use sample::InputAndTargets;
use sample_file::{SampleFile, SampleFileReader};
use self_play::SelfPlayMetrics;
use serde::{de, Serialize};
use std::assert_matches::assert_matches;
use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use numpy::IntoPyArray;
use pyo3::{exceptions::PyFileNotFoundError, prelude::*};

mod arimaa_sampler;
mod dir_index;
mod index;
mod sample;
mod sample_file;

use arimaa_sampler::ArimaaSampler as Sampler;

use crate::index::*;
use crate::sample::*;

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn replay_buffer(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<ReplayBuffer>()?;
    Ok(())
}

#[pyclass]
struct ReplayBuffer {
    index: Index,
    sample_loader: SampleLoader<Sampler>,
}

#[pymethods]
impl ReplayBuffer {
    #[new]
    fn new(
        games_dir: String,
        min_visits: usize,
        mode: Option<String>,
        cache_dir: Option<String>,
    ) -> PyResult<Self> {
        env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();

        let cache_dir = cache_dir.unwrap_or_else(|| "buffer_cache".to_string());
        let cache_dir = PathBuf::from(cache_dir);

        let sampler = Sampler::new(mode);
        let input_size = sampler.input_size();
        let policy_size = sampler.policy_size();
        let moves_left_size = sampler.moves_left_size();
        let num_values_in_sample = input_size + policy_size + 1 + moves_left_size;
        let index_res = Index::new(PathBuf::from(games_dir));
        let index = index_res
            .map_err(|_| PyErr::new::<PyFileNotFoundError, _>("Failed to index game files."))?;

        Ok(Self {
            index,
            sample_loader: SampleLoader {
                cache_dir,
                min_visits,
                num_values_in_sample,
                sampler,
            },
        })
    }

    fn sample<'py>(
        &mut self,
        py: Python<'py>,
        samples: usize,
        start_idx: usize,
        end_idx: usize,
    ) -> PyResult<&'py PyDict> {
        let num_samples = samples;
        let sampler = &self.sample_loader.sampler;

        let failures = AtomicUsize::new(0);
        let path_sampler = &self.index.sampler(start_idx..end_idx).map_err(|_| {
            PyErr::new::<PyFileNotFoundError, _>("Failed to index during sampling.")
        })?;

        let samples: Vec<_> = (0..samples)
            .into_par_iter()
            .map(|_| {
                loop {
                    let sample_path = path_sampler.sample();

                    if let Ok(sample) = self.sample_loader.load_and_sample_metrics(&sample_path) {
                        if let Some(sample) = sample {
                            return sample;
                        } else {
                            // No samples entries were found for that specific game. Try another.
                            continue;
                        }
                    }

                    let num_failures = failures.fetch_add(1, Ordering::SeqCst);

                    if num_failures > samples {
                        panic!("Failed too many times attempting to load samples");
                    }
                }
            })
            .collect();

        let mut x = Vec::with_capacity(num_samples * sampler.input_size());
        let mut yp = Vec::with_capacity(num_samples * sampler.policy_size());
        let mut yv = Vec::with_capacity(num_samples);
        let mut ym = Vec::with_capacity(num_samples * sampler.moves_left_size());

        for input_outputs in samples {
            x.extend_from_slice(&input_outputs.input);
            yp.extend_from_slice(&input_outputs.policy_output);
            yv.push(input_outputs.value_output);
            ym.extend_from_slice(&input_outputs.moves_left_output);
        }

        let dict = Ok(HashMap::from([
            ("X", x.into_pyarray(py)),
            ("yp", yp.into_pyarray(py)),
            ("yv", yv.into_pyarray(py)),
            ("ym", ym.into_pyarray(py)),
        ])
        .into_py_dict(py));

        dict
    }

    fn games(&mut self) -> PyResult<usize> {
        self.re_index()
            .map_err(|_| PyErr::new::<PyFileNotFoundError, _>("Failed to index game files."))?;

        Ok(self.index.games())
    }

    fn avg_num_samples_per_game(&mut self, look_back: usize) -> f32 {
        let num_samples = self
            .index
            .iter()
            .rev()
            .take(look_back)
            .par_bridge()
            .filter_map(|path| {
                let mut sampler = self.sample_loader.load_and_cache_samples(path).ok()?;
                sampler.num_samples().ok()
            })
            .collect::<Vec<_>>();

        num_samples.into_iter().zip(1..).fold(0., |s, (e, i)| {
            (e as f32 + s * (i as f32 - 1.0) as f32) / i as f32
        })
    }
}

impl ReplayBuffer {
    fn re_index(&mut self) -> Result<()> {
        self.index.re_index()
    }
}

struct SampleLoader<S> {
    cache_dir: PathBuf,
    min_visits: usize,
    num_values_in_sample: usize,
    sampler: S,
}

impl<S> SampleLoader<S> {
    fn load_and_sample_metrics(
        &self,
        metrics_path: impl AsRef<Path>,
    ) -> Result<Option<InputAndTargets>>
    where
        S: Sample,
        S::State: GameState,
        S::Action: de::DeserializeOwned + Serialize,
        S::Value: de::DeserializeOwned + Serialize + Clone,
    {
        let input_size = self.sampler.input_size();
        let policy_size = self.sampler.policy_size();
        let moves_left_size = self.sampler.moves_left_size();

        let mut sample_reader = self.load_and_cache_samples(metrics_path)?;

        let num_samples = sample_reader.num_samples()?;

        if num_samples == 0 {
            return Ok(None);
        }

        let rand_sample_idx = rand::thread_rng().gen_range(0..num_samples);

        let mut vals = sample_reader.read_sample(rand_sample_idx)?.into_iter();

        let inputs_and_targets = InputAndTargets {
            input: vals.by_ref().take(input_size).collect(),
            policy_output: vals.by_ref().take(policy_size).collect(),
            value_output: vals.next().unwrap(),
            moves_left_output: vals.by_ref().take(moves_left_size).collect(),
        };

        assert_matches!(vals.next(), None, "No more vals should be left");

        Ok(Some(inputs_and_targets))
    }

    fn load_and_cache_samples(
        &self,
        metrics_path: impl AsRef<Path>,
    ) -> Result<SampleFileReader<BufReader<File>>>
    where
        S: Sample,
        S::State: GameState,
        S::Action: de::DeserializeOwned + Serialize,
        S::Value: de::DeserializeOwned + Serialize + Clone,
    {
        let cache_path = &self.metrics_path_for_cache(&metrics_path)?;

        let read_cache_file = move || -> Result<SampleFileReader<BufReader<File>>> {
            let file = File::open(cache_path)?;

            let res = try {
                let buff_reader = BufReader::new(file);
                SampleFile::new(self.num_values_in_sample).read(buff_reader)
            };

            if matches!(res, Err(_)) {
                println!("Failed to read {:?}", cache_path);
            }

            res
        };

        let mut buffered_samples = read_cache_file();

        if matches!(buffered_samples, Err(_)) {
            let res: Result<usize> = try {
                fs::create_dir_all(&cache_path.parent().unwrap())?;

                let file = OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(cache_path)?;

                let samples = self.load_samples(metrics_path)?;
                let mut vals = Vec::with_capacity(samples.len() * self.num_values_in_sample);
                for metrics in samples {
                    let inputs_and_targets = self.sampler.metric_to_input_and_targets(&metrics);
                    vals.extend(inputs_and_targets.input);
                    vals.extend(inputs_and_targets.policy_output);
                    vals.push(inputs_and_targets.value_output);
                    vals.extend(inputs_and_targets.moves_left_output);
                }

                let buff_writer = BufWriter::new(file);
                SampleFile::new(self.num_values_in_sample).write(buff_writer, &vals)?;

                buffered_samples = read_cache_file();

                0
            };

            if let Err(err) = res {
                warn!("Failed to write cache file {:?} {:?}", cache_path, err);
            }
        }

        buffered_samples
    }

    #[allow(clippy::type_complexity)]
    fn load_samples(
        &self,
        metrics_path: impl AsRef<Path>,
    ) -> Result<Vec<PositionMetrics<S::State, S::Action, S::Value>>>
    where
        S: Sample,
        S::State: GameState,
        S::Action: de::DeserializeOwned + Serialize,
        S::Value: de::DeserializeOwned + Serialize + Clone,
    {
        let file = std::fs::File::open(&metrics_path)
            .with_context(|| format!("Failed to open: {:?}", &metrics_path.as_ref()))?;
        let file = GzDecoder::new(file);
        let metrics: SelfPlayMetrics<<S as Sample>::Action, <S as Sample>::Value> =
            serde_json::from_reader(file)
                .with_context(|| format!("Failed to deserialize: {:?}", &metrics_path.as_ref()))?;

        let samples = self.sampler.metrics_to_samples(metrics, self.min_visits);

        Ok(samples)
    }

    fn metrics_path_for_cache(&self, metrics_path: impl AsRef<Path>) -> Result<PathBuf> {
        let mut components = metrics_path.as_ref().components().collect::<Vec<_>>();

        components.reverse();

        let mut components = components.into_iter();

        let metrics_file_name = components.next().unwrap().as_os_str();
        let model_dir = components.next().unwrap().as_os_str();

        Ok(self
            .cache_dir
            .to_path_buf()
            .join(model_dir)
            .join(metrics_file_name))
    }
}
