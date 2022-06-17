#![feature(try_blocks)]
#![feature(let_chains)]
#![feature(assert_matches)]
#![feature(test)]
#![feature(bench_black_box)]

use anyhow::{Context, Result};
use engine::GameState;
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
use std::fs::{self, File};
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use numpy::IntoPyArray;
use pyo3::{exceptions::PyFileNotFoundError, prelude::*};

mod arimaa_sampler;
mod index;
mod sample;
mod sample_file;

use arimaa_sampler::ArimaaSampler as Sampler;

use crate::index::Index;
use crate::sample::Sample;

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
    min_visits: usize,
    index: Index,
    sampler: Sampler,
    cache_dir: PathBuf
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
        let cache_dir = cache_dir.unwrap_or_else(|| "buffer_cache".to_string());
        let cache_dir = PathBuf::from(cache_dir);

        Ok(Self {
            index: Index::new(PathBuf::from(games_dir))
                .map_err(|_| PyErr::new::<PyFileNotFoundError, _>("Failed to index game files."))?,
            min_visits,
            cache_dir,
            sampler: Sampler::new(mode),
        })
    }

    fn sample<'py>(&self, py: Python<'py>, samples: usize, start_idx: usize, end_idx: usize) -> PyResult<&'py PyDict> {
        let num_samples = samples;
        let min_visits = self.min_visits;

        let failures = AtomicUsize::new(0);

        let samples: Vec<_> = (0..samples)
            .into_par_iter()
            .map(|_| {
                loop {
                    let sample_path = self.index.sample(start_idx, end_idx);

                    if let Ok(sample) = load_and_sample_metrics(&sample_path, &self.cache_dir, min_visits, &self.sampler) {
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

        let mut x = Vec::with_capacity(num_samples * self.sampler.input_size());
        let mut yp = Vec::with_capacity(num_samples * self.sampler.policy_size());
        let mut yv = Vec::with_capacity(num_samples);
        let mut ym = Vec::with_capacity(num_samples * self.sampler.moves_left_size());

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

    fn games(&self) -> usize {
        self.index.games()
    }
}

fn load_and_sample_metrics<S: Sample>(
    metrics_path: impl AsRef<Path>,
    cache_dir: impl AsRef<Path>,
    min_visits: usize,
    sampler: &S,
) -> Result<Option<InputAndTargets>>
where
    S::State: GameState,
    S::Action: de::DeserializeOwned + Serialize,
    S::Value: de::DeserializeOwned + Serialize + Clone,
{
    let input_size = sampler.input_size();
    let policy_size = sampler.policy_size();
    let moves_left_size = sampler.moves_left_size();
    let num_values_in_sample = input_size + policy_size + 1 + moves_left_size;

    let mut sample_reader = load_and_cache_samples(
        metrics_path,
        cache_dir,
        min_visits,
        num_values_in_sample,
        sampler,
    )?;

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

fn load_and_cache_samples<S: Sample>(
    metrics_path: impl AsRef<Path>,
    cache_dir: impl AsRef<Path>,
    min_visits: usize,
    num_values_in_sample: usize,
    sampler: &S,
) -> Result<SampleFileReader<BufReader<File>>>
where
    S::State: GameState,
    S::Action: de::DeserializeOwned + Serialize,
    S::Value: de::DeserializeOwned + Serialize + Clone,
{
    let cache_path = &metrics_path_for_cache(&metrics_path, cache_dir)?;

    let read_cache_file = move || -> Result<SampleFileReader<BufReader<File>>> {
        let file = File::open(cache_path)?;

        let res = try {
            let buff_reader = BufReader::new(file);
            SampleFile::new(num_values_in_sample).read(buff_reader)
        };

        if matches!(res, Err(_)) {
            println!("Failed to read {:?}", cache_path);
        }

        res
    };

    let mut buffered_samples = read_cache_file();

    if matches!(buffered_samples, Err(_)) {
        let samples = load_samples(metrics_path, min_visits, sampler)?;
        let mut vals = Vec::with_capacity(samples.len() * num_values_in_sample);
        for metrics in samples {
            let inputs_and_targets = sampler.metric_to_input_and_targets(&metrics);
            vals.extend(inputs_and_targets.input);
            vals.extend(inputs_and_targets.policy_output);
            vals.push(inputs_and_targets.value_output);
            vals.extend(inputs_and_targets.moves_left_output);
        }

        if !cache_path.exists() {
            fs::create_dir_all(&cache_path.parent().unwrap())?;

            let file = File::create(cache_path)?;
            let buff_writer = BufWriter::new(file);
            SampleFile::new(num_values_in_sample).write(buff_writer, &vals)?;
        } else {
            warn!("Path exits but was not loaded {:?}", cache_path);
        }

        buffered_samples = read_cache_file();
    }

    buffered_samples
}

#[allow(clippy::type_complexity)]
fn load_samples<S: Sample>(
    metrics_path: impl AsRef<Path>,
    min_visits: usize,
    sampler: &S,
) -> Result<Vec<PositionMetrics<S::State, S::Action, S::Value>>>
where
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

    let samples = sampler.metrics_to_samples(metrics, min_visits);

    Ok(samples)
}

// input 4 + 1
// output 70
// moves left 1

fn metrics_path_for_cache(
    metrics_path: impl AsRef<Path>,
    cache_dir: impl AsRef<Path>,
) -> Result<PathBuf> {
    let mut components = metrics_path.as_ref().components().collect::<Vec<_>>();

    components.reverse();

    let mut components = components.into_iter();

    let metrics_file_name = components.next().unwrap().as_os_str();
    let model_dir = components.next().unwrap().as_os_str();

    Ok(cache_dir
        .as_ref()
        .to_path_buf()
        .join(model_dir)
        .join(metrics_file_name))
}
