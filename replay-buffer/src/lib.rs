use anyhow::{Context, Result};
use common::PropagatedValue;
use engine::GameState;
use env_logger::Env;
use flate2::read::GzDecoder;
use log::warn;
use pyo3::types::{IntoPyDict, PyDict};
use q_mix::{PredictionStore, QMix};
use rand::Rng;
use rayon::prelude::*;
use sample::InputAndTargets;
use sample_file::{SampleFile, SampleFileReader};
use self_play::SelfPlayMetrics;
use serde::{de, Serialize};
use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use tensorflow_model::{InputMap, PredictionsMap};

use numpy::IntoPyArray;
use pyo3::{exceptions::PyFileNotFoundError, prelude::*};

mod arimaa_sampler;
mod deblunder;
mod deblunder_test;
mod dir_index;
mod index;
mod q_mix;
mod quoridor_sampler;
mod sample;
mod sample_file;

use quoridor_sampler::QuoridorSampler as Sampler;

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
        q_diff_threshold: f32,
        q_diff_width: f32,
        mode: Option<String>,
        cache_dir: Option<String>,
    ) -> PyResult<Self> {
        env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();

        let cache_dir = cache_dir.unwrap_or_else(|| "buffer_cache".to_string());
        let cache_dir = PathBuf::from(cache_dir);

        let sampler = Sampler::new(mode);
        let num_values_in_sample = sampler.sample_size();
        let index_res = Index::new(PathBuf::from(games_dir));
        let index = index_res
            .map_err(|_| PyErr::new::<PyFileNotFoundError, _>("Failed to index game files."))?;

        Ok(Self {
            index,
            sample_loader: SampleLoader {
                cache_dir,
                min_visits,
                q_diff_threshold,
                q_diff_width,
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

                    let num_failures: usize = failures.fetch_add(1, Ordering::SeqCst);

                    if num_failures > samples {
                        panic!("Failed too many times attempting to load samples");
                    }
                }
            })
            .collect();

        let outputs = sampler.outputs();
        let mut data: HashMap<String, Vec<f32>> = outputs
            .iter()
            .map(|(key, size)| (key.to_owned(), Vec::with_capacity(num_samples * size)))
            .collect();

        for sample in samples.iter() {
            for (name, _) in outputs.iter() {
                let values = sampler.output_values(sample, name);
                data.get_mut(name)
                    .expect("No output entry found.")
                    .extend_from_slice(values);
            }
        }

        for (key, target) in data.iter() {
            let expected_size = sampler.output_offset_and_size(key).1 * num_samples;
            let actual_size = target.len();
            assert_eq!(
                expected_size, actual_size,
                "Samples must be fully filled with values. Expected size: {}, Actual size: {}",
                expected_size, actual_size
            );
        }

        let mut inputs = Vec::with_capacity(num_samples * sampler.input_size());
        for sample in samples {
            let input: &[f32] = sampler.input_values(&sample);
            inputs.extend_from_slice(input);
        }

        data.insert("inputs".to_string(), inputs);

        let dict = data
            .into_iter()
            .map(|(k, v)| (k, v.into_pyarray(py)))
            .collect::<HashMap<_, _>>()
            .into_py_dict(py);

        Ok(dict)
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

        num_samples
            .into_iter()
            .zip(1..)
            .fold(0., |s, (e, i)| (e as f32 + s * (i as f32 - 1.0)) / i as f32)
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
    q_diff_threshold: f32,
    q_diff_width: f32,
}

impl<S> SampleLoader<S> {
    fn load_and_sample_metrics(
        &self,
        metrics_path: impl AsRef<Path>,
    ) -> Result<Option<InputAndTargets>>
    where
        S: Sample,
        S: Sized,
        <S as Sample>::State: GameState,
        <S as Sample>::Action: de::DeserializeOwned + Serialize + PartialEq + Clone,
        <S as Sample>::Predictions: de::DeserializeOwned + Serialize + Clone,
        <S as Sample>::PropagatedValues: PropagatedValue + de::DeserializeOwned + Serialize,
        S: InputMap<State = <S as Sample>::State>,
        S: PredictionsMap<
            State = <S as Sample>::State,
            Action = <S as Sample>::Action,
            Predictions = <S as Sample>::Predictions,
            PropagatedValues = <S as Sample>::PropagatedValues,
        >,
        S: Sample,
        <S as Sample>::State: GameState,
        <S as Sample>::Action: de::DeserializeOwned + Serialize + PartialEq,
        <S as Sample>::Predictions: de::DeserializeOwned + Serialize + Clone,
        <S as Sample>::PropagatedValues: PropagatedValue + de::DeserializeOwned + Serialize,
        S::PredictionStore:
            PredictionStore<State = <S as Sample>::State, Predictions = <S as Sample>::Predictions>,
        S: QMix<
            State = <S as Sample>::State,
            Predictions = <S as Sample>::Predictions,
            PropagatedValues = <S as Sample>::PropagatedValues,
        >,
    {
        let mut sample_reader = self.load_and_cache_samples(metrics_path)?;

        let num_samples = sample_reader.num_samples()?;

        if num_samples == 0 {
            return Ok(None);
        }

        let rand_sample_idx = rand::thread_rng().gen_range(0..num_samples);

        let inputs_and_targets = sample_reader.read_sample(rand_sample_idx)?.into();

        Ok(Some(inputs_and_targets))
    }

    fn load_and_cache_samples(
        &self,
        metrics_path: impl AsRef<Path>,
    ) -> Result<SampleFileReader<BufReader<File>>>
    where
        S: Sample,
        S: Sized,
        <S as Sample>::State: GameState,
        <S as Sample>::Action: de::DeserializeOwned + Serialize + PartialEq + Clone,
        <S as Sample>::Predictions: de::DeserializeOwned + Serialize + Clone,
        <S as Sample>::PropagatedValues: PropagatedValue + de::DeserializeOwned + Serialize,
        S: InputMap<State = <S as Sample>::State>,
        S: PredictionsMap<
            State = <S as Sample>::State,
            Action = <S as Sample>::Action,
            Predictions = <S as Sample>::Predictions,
            PropagatedValues = <S as Sample>::PropagatedValues,
        >,
        S: Sample,
        <S as Sample>::State: GameState,
        <S as Sample>::Action: de::DeserializeOwned + Serialize + PartialEq,
        <S as Sample>::Predictions: de::DeserializeOwned + Serialize + Clone,
        <S as Sample>::PropagatedValues: PropagatedValue + de::DeserializeOwned + Serialize,
        S::PredictionStore:
            PredictionStore<State = <S as Sample>::State, Predictions = <S as Sample>::Predictions>,
        S: QMix<
            State = <S as Sample>::State,
            Predictions = <S as Sample>::Predictions,
            PropagatedValues = <S as Sample>::PropagatedValues,
        >,
    {
        let cache_path = &self.metrics_path_for_cache(&metrics_path)?;

        let read_cache_file = move || -> Result<SampleFileReader<BufReader<File>>> {
            let file = File::open(cache_path)?;

            let res = {
                let buff_reader = BufReader::new(file);
                SampleFile::new(self.num_values_in_sample).read(buff_reader)
            };

            Ok(res)
        };

        let mut buffered_samples = read_cache_file();

        if buffered_samples.is_err() {
            let res: Result<usize> = (|| {
                fs::create_dir_all(cache_path.parent().unwrap())?;

                let file = OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(cache_path)?;

                let samples = self.load_samples(metrics_path)?;
                let mut vals = Vec::with_capacity(samples.len() * self.num_values_in_sample);
                for sample in samples {
                    let inputs_and_targets = self.sampler.metric_to_input_and_targets(sample.target_score, &sample.metrics);
                    vals.extend_from_slice(inputs_and_targets.as_slice());
                }

                let buff_writer = BufWriter::new(file);
                SampleFile::new(self.num_values_in_sample).write(buff_writer, &vals)?;

                buffered_samples = read_cache_file();

                Ok(0)
            })();

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
    ) -> Result<
        Vec<
            PositionMetricsExtended<
                <S as Sample>::State,
                <S as Sample>::Action,
                <S as Sample>::Predictions,
                <S as Sample>::PropagatedValues,
            >,
        >,
    >
    where
        S: Sample,
        S: Sized,
        <S as Sample>::State: GameState,
        <S as Sample>::Action: de::DeserializeOwned + Serialize + PartialEq + Clone,
        <S as Sample>::Predictions: de::DeserializeOwned + Serialize + Clone,
        <S as Sample>::PropagatedValues: PropagatedValue + de::DeserializeOwned + Serialize,
        S: InputMap<State = <S as Sample>::State>,
        S: PredictionsMap<
            State = <S as Sample>::State,
            Action = <S as Sample>::Action,
            Predictions = <S as Sample>::Predictions,
            PropagatedValues = <S as Sample>::PropagatedValues,
        >,
        S::PredictionStore:
            PredictionStore<State = <S as Sample>::State, Predictions = <S as Sample>::Predictions>,
        S: QMix<
            State = <S as Sample>::State,
            Predictions = <S as Sample>::Predictions,
            PropagatedValues = <S as Sample>::PropagatedValues,
        >,
    {
        let file = std::fs::File::open(&metrics_path)
            .with_context(|| format!("Failed to open: {:?}", &metrics_path.as_ref()))?;
        let file = GzDecoder::new(file);
        let metrics: SelfPlayMetrics<
            <S as Sample>::Action,
            <S as Sample>::Predictions,
            <S as Sample>::PropagatedValues,
        > = serde_json::from_reader(file)
            .with_context(|| format!("Failed to deserialize: {:?}", &metrics_path.as_ref()))?;

        let samples = self.sampler.metrics_to_samples(
            metrics,
            self.min_visits,
            self.q_diff_threshold,
            self.q_diff_width,
        );

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
