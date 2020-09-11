use anyhow::{anyhow, Context as AnyhowContext, Result};
use common::chunk_queue::ConcurrentChunkQueue;
use common::incrementing_map::IncrementingMap;
use engine::engine::GameEngine;
use engine::game_state::GameState;
use engine::value::Value;
use half::f16;
use itertools::Itertools;
use log::info;
use parking_lot::Mutex;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use std::collections::binary_heap::PeekMut;
use std::collections::BinaryHeap;
use std::fs::{self, File};
use std::future::Future;
use std::hash::Hash;
use std::io::{BufReader, Write};
use std::path::PathBuf;
use std::pin::Pin;
use std::process::{Command, Stdio};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use std::task::{Context, Poll, Waker};
use tensorflow::{Graph, Operation, Session, SessionOptions, SessionRunArgs, Tensor};

use super::super::analytics::{self, GameStateAnalysis};
use super::super::model::{Model as ModelTrait, TrainOptions};
use super::super::model_info::ModelInfo;
use super::super::node_metrics::NodeMetrics;
use super::super::position_metrics::PositionMetrics;
use super::constants::{
    ANALYSIS_REQUEST_BATCH_SIZE, ANALYSIS_REQUEST_THREADS, TRAIN_DATA_CHUNK_SIZE,
};
use super::mode::Mode;
use super::paths::Paths;
use super::reporter::Reporter;
use super::transposition_table::TranspositionTable;

#[cfg_attr(feature = "tensorflow_system_alloc", global_allocator)]
#[cfg(feature = "tensorflow_system_alloc")]
static ALLOCATOR: std::alloc::System = std::alloc::System;

pub struct TensorflowModel<S, A, V, E, Map, Te> {
    model_info: ModelInfo,
    batching_model: Arc<BatchingModel<S, A, V, E, Map, Te>>,
    active_analyzers: Arc<AtomicUsize>,
    active_threads: Arc<AtomicUsize>,
    mapper: Arc<Map>,
    analysis_request_threads: usize,
    options: TensorflowModelOptions,
    batch_size: usize,
}

#[derive(Serialize, Deserialize)]
pub struct TensorflowModelOptions {
    pub num_filters: usize,
    pub num_blocks: usize,
    pub channel_height: usize,
    pub channel_width: usize,
    pub channels: usize,
    pub output_size: usize,
    pub moves_left_size: usize,
}

pub trait Mapper<S, A, V, Te> {
    fn game_state_to_input(&self, game_state: &S, mode: Mode) -> Vec<half::f16>;
    fn get_input_dimensions(&self) -> [u64; 3];
    fn get_symmetries(&self, metric: PositionMetrics<S, A, V>) -> Vec<PositionMetrics<S, A, V>>;
    fn policy_metrics_to_expected_output(
        &self,
        game_state: &S,
        policy: &NodeMetrics<A>,
    ) -> Vec<f32>;
    fn map_value_to_value_output(&self, game_state: &S, value: &V) -> f32;
    fn map_output_to_transposition_entry<I: Iterator<Item = f16>>(
        &self,
        game_state: &S,
        policy_scores: I,
        value: f16,
        moves_left: f32,
    ) -> Te;
    fn map_transposition_entry_to_analysis(
        &self,
        game_state: &S,
        transposition_entry: &Te,
    ) -> GameStateAnalysis<A, V>;
    fn get_transposition_key(&self, game_state: &S) -> u64;
}

impl<S, A, V, E, Map, Te> TensorflowModel<S, A, V, E, Map, Te>
where
    S: PartialEq + Hash + Send + Sync + 'static,
    A: Clone + Send + Sync + 'static,
    V: Value + Send + Sync + 'static,
    E: GameEngine<State = S, Action = A, Value = V> + Send + Sync + 'static,
    Map: Mapper<S, A, V, Te> + Send + Sync + 'static,
    Te: Send + Sync + 'static,
{
    pub fn new(model_info: ModelInfo, engine: E, mapper: Map, tt_cache_size: usize) -> Self {
        let analysis_request_threads = std::env::var("ANALYSIS_REQUEST_THREADS")
            .map(|v| {
                v.parse::<usize>()
                    .expect("ANALYSIS_REQUEST_THREADS must be a valid int")
            })
            .unwrap_or(ANALYSIS_REQUEST_THREADS);

        let batch_size = std::env::var("ANALYSIS_REQUEST_BATCH_SIZE")
            .map(|v| {
                v.parse::<usize>()
                    .expect("ANALYSIS_REQUEST_BATCH_SIZE must be a valid int")
            })
            .unwrap_or(ANALYSIS_REQUEST_BATCH_SIZE);

        if std::env::var("TF_CPP_MIN_LOG_LEVEL").is_err() {
            std::env::set_var("TF_CPP_MIN_LOG_LEVEL", "2");
        }

        let mapper = Arc::new(mapper);
        let transposition_table = Arc::new(if tt_cache_size > 0 {
            Some(TranspositionTable::new(tt_cache_size))
        } else {
            None
        });

        let reporter = Arc::new(Reporter::new(transposition_table.clone()));

        let batching_model = Arc::new(BatchingModel::new(
            engine,
            mapper.clone(),
            transposition_table,
            reporter,
            analysis_request_threads,
            batch_size,
        ));
        let options = get_options(&model_info).expect("Could not load model options file");

        Self {
            batch_size,
            model_info,
            batching_model,
            active_analyzers: Arc::new(AtomicUsize::new(0)),
            active_threads: Arc::new(AtomicUsize::new(0)),
            mapper,
            analysis_request_threads,
            options,
        }
    }

    pub fn create(model_info: &ModelInfo, options: &TensorflowModelOptions) -> Result<()> {
        create(model_info, options)
    }

    #[allow(clippy::too_many_arguments)]
    fn create_analysis_threads(
        active_threads: &Arc<AtomicUsize>,
        active_analyzers: &Arc<AtomicUsize>,
        batching_model: &Arc<BatchingModel<S, A, V, E, Map, Te>>,
        model_info: &ModelInfo,
        mapper: &Arc<Map>,
        analysis_request_threads: usize,
        output_size: usize,
        moves_left_size: usize,
        batch_size: usize,
    ) {
        loop {
            let thread_num = active_threads.fetch_add(1, Ordering::Relaxed);

            if thread_num >= analysis_request_threads {
                active_threads.fetch_sub(1, Ordering::Relaxed);
                break;
            }

            let batch_ref = batching_model.clone();
            let model_info = model_info.clone();
            let mapper = mapper.clone();
            let active_analyzers = active_analyzers.clone();
            let active_threads = active_threads.clone();

            tokio::task::spawn_blocking(move || {
                let input_dim = mapper.get_input_dimensions();
                let predictor = Predictor::new(&model_info, input_dim);
                loop {
                    let mut states_to_analyse = Vec::with_capacity(batch_size);
                    let mut analyzed_states = Vec::with_capacity(batch_size);

                    for (id, state_to_analyse, waker) in batch_ref.states_to_analyse.draining_iter()
                    {
                        match batch_ref.try_immediate_analysis(&state_to_analyse) {
                            Some(analysis) => analyzed_states.push((id, analysis, waker)),
                            None => states_to_analyse.push((id, state_to_analyse, waker)),
                        }

                        if states_to_analyse.len() == batch_size {
                            break;
                        }
                    }

                    if states_to_analyse.is_empty() && analyzed_states.is_empty() {
                        if active_analyzers.load(Ordering::Relaxed) == 0 {
                            break;
                        }

                        std::thread::sleep(std::time::Duration::from_millis(1));
                    }

                    if !analyzed_states.is_empty() {
                        let batch_ref_clone = batch_ref.clone();
                        tokio::task::spawn_blocking(move || {
                            batch_ref_clone
                                .states_analysed
                                .wake_with_analysis(analyzed_states.into_iter());
                        });
                    }

                    if !states_to_analyse.is_empty() {
                        let game_states_to_predict =
                            states_to_analyse.iter().map(|(_, game_state, _)| {
                                mapper.game_state_to_input(game_state, Mode::Infer)
                            });

                        let predictions = predictor
                            .predict(game_states_to_predict, states_to_analyse.len())
                            .expect("Expected predict to be successful");

                        let batch_ref_clone = batch_ref.clone();
                        tokio::task::spawn_blocking(move || {
                            let results = batch_ref_clone.collect_analysis(
                                states_to_analyse.into_iter(),
                                predictions,
                                output_size,
                                moves_left_size,
                            );

                            batch_ref_clone
                                .states_analysed
                                .wake_with_analysis(results.into_iter());
                        });
                    }
                }

                active_threads.fetch_sub(1, Ordering::Relaxed);
            });
        }
    }
}

impl<S, A, V, E, Map, Te> ModelTrait for TensorflowModel<S, A, V, E, Map, Te>
where
    S: GameState + Send + Sync + Unpin + 'static,
    A: Clone + Send + Sync + 'static,
    V: Value + Send + Sync + 'static,
    E: GameEngine<State = S, Action = A, Value = V> + Send + Sync + 'static,
    Map: Mapper<S, A, V, Te> + Send + Sync + 'static,
    Te: Send + Sync + 'static,
{
    type State = S;
    type Action = A;
    type Value = V;
    type Analyzer = GameAnalyzer<S, A, V, E, Map, Te>;

    fn get_model_info(&self) -> &ModelInfo {
        &self.model_info
    }

    fn train<I>(
        &self,
        target_model_info: &ModelInfo,
        sample_metrics: I,
        options: &TrainOptions,
    ) -> Result<()>
    where
        I: Iterator<Item = PositionMetrics<S, A, V>>,
    {
        train(
            &self.model_info,
            target_model_info,
            sample_metrics,
            self.mapper.clone(),
            options,
        )
    }

    fn get_game_state_analyzer(&self) -> Self::Analyzer {
        self.active_analyzers.fetch_add(1, Ordering::Relaxed);

        Self::create_analysis_threads(
            &self.active_threads,
            &self.active_analyzers,
            &self.batching_model,
            &self.model_info,
            &self.mapper,
            self.analysis_request_threads,
            self.options.output_size,
            self.options.moves_left_size,
            self.batch_size,
        );

        Self::Analyzer {
            batching_model: self.batching_model.clone(),
            active_analyzers: self.active_analyzers.clone(),
        }
    }
}

struct Predictor {
    session: SessionAndOps,
    input_dimensions: [u64; 3],
}

impl Predictor {
    fn new(model_info: &ModelInfo, input_dimensions: [u64; 3]) -> Self {
        let mut graph = Graph::new();

        let exported_model_path = format!(
            "{game_name}_runs/{run_name}/tensorrt_models/{model_num}",
            game_name = model_info.get_game_name(),
            run_name = model_info.get_run_name(),
            model_num = model_info.get_model_num(),
        );

        let exported_model_path = std::env::current_dir()
            .expect("Expected to be able to open the model directory")
            .join(exported_model_path);

        info!("{:?}", exported_model_path);

        let session = Session::from_saved_model(
            &SessionOptions::new(),
            &["serve"],
            &mut graph,
            exported_model_path,
        )
        .expect("Expected to be able to load model");

        let op_input = graph
            .operation_by_name_required("input_1")
            .expect("Expected to find input operation");
        let op_value_head = graph
            .operation_by_name_required("value_head/Tanh")
            .expect("Expected to find value_head operation");
        let op_policy_head = graph
            .operation_by_name_required("policy_head/concat")
            .or_else(|_| graph.operation_by_name_required("policy_head/BiasAdd"))
            .expect("Expected to find policy_head operation");
        let op_moves_left_head = graph
            .operation_by_name_required("moves_left_head/Softmax")
            .ok();

        Self {
            input_dimensions,
            session: SessionAndOps {
                session,
                op_input,
                op_value_head,
                op_policy_head,
                op_moves_left_head,
            },
        }
    }

    fn predict<I: Iterator<Item = Vec<f16>>>(
        &self,
        game_state_inputs: I,
        batch_size: usize,
    ) -> Result<AnalysisResults> {
        let input_dim = self.input_dimensions;
        let input_dimensions = [batch_size as u64, input_dim[0], input_dim[1], input_dim[2]];

        let mut input_tensor: Tensor<f16> = Tensor::new(&input_dimensions);
        fill_tensor(&mut input_tensor, game_state_inputs.flatten());

        let session = &self.session;

        let mut output_step = SessionRunArgs::new();
        output_step.add_feed(&session.op_input, 0, &input_tensor);
        let value_head_fetch_token = output_step.request_fetch(&session.op_value_head, 0);
        let policy_head_fetch_token = output_step.request_fetch(&session.op_policy_head, 0);
        let moves_left_head_fetch_token = session
            .op_moves_left_head
            .as_ref()
            .map(|op| output_step.request_fetch(op, 0));

        session
            .session
            .run(&mut output_step)
            .expect("Expected to be able to run the model session");

        let value_head_output: Tensor<f16> = output_step
            .fetch(value_head_fetch_token)
            .expect("Expected to be able to load value_head output");
        let policy_head_output: Tensor<f16> = output_step
            .fetch(policy_head_fetch_token)
            .expect("Expected to be able to load policy_head output");
        let moves_left_head_output: Option<Tensor<f16>> =
            moves_left_head_fetch_token.map(|moves_left_head_fetch_token| {
                output_step
                    .fetch(moves_left_head_fetch_token)
                    .expect("Expected to be able to load moves_left_head output")
            });

        Ok(AnalysisResults {
            policy_head_output,
            value_head_output,
            moves_left_head_output,
        })
    }
}

struct AnalysisResults {
    policy_head_output: Tensor<f16>,
    value_head_output: Tensor<f16>,
    moves_left_head_output: Option<Tensor<f16>>,
}

struct SessionAndOps {
    session: Session,
    op_input: Operation,
    op_value_head: Operation,
    op_policy_head: Operation,
    op_moves_left_head: Option<Operation>,
}

pub struct GameAnalyzer<S, A, V, E, Map, Te> {
    batching_model: Arc<BatchingModel<S, A, V, E, Map, Te>>,
    active_analyzers: Arc<AtomicUsize>,
}

impl<S, A, V, E, Map, Te> analytics::GameAnalyzer for GameAnalyzer<S, A, V, E, Map, Te>
where
    S: Clone + PartialEq + Hash + Unpin,
    A: Clone,
    V: Value,
    E: GameEngine<State = S, Action = A, Value = V> + Send + Sync + 'static,
    Map: Mapper<S, A, V, Te>,
    Te: Send + Sync + 'static,
{
    type State = S;
    type Action = A;
    type Value = V;
    type Future = GameStateAnalysisFuture<S, A, V, E, Map, Te>;

    /// Outputs a value from [-1, 1] depending on the player to move's evaluation of the current state.
    /// If the evaluation is a draw then 0.0 will be returned.
    /// Along with the value output a list of policy scores for all VALID moves is returned. If the position
    /// is terminal then the vector will be empty.
    fn get_state_analysis(&self, game_state: &S) -> GameStateAnalysisFuture<S, A, V, E, Map, Te> {
        GameStateAnalysisFuture::new(game_state.to_owned(), self.batching_model.clone())
    }
}

impl<S, A, V, E, Map, Te> Drop for GameAnalyzer<S, A, V, E, Map, Te> {
    fn drop(&mut self) {
        self.active_analyzers.fetch_sub(1, Ordering::Relaxed);
    }
}

#[allow(non_snake_case)]
fn train<S, A, V, I, Map, Te>(
    source_model_info: &ModelInfo,
    target_model_info: &ModelInfo,
    sample_metrics: I,
    mapper: Arc<Map>,
    options: &TrainOptions,
) -> Result<()>
where
    S: GameState + Send + Sync + 'static,
    A: Clone + Send + Sync + 'static,
    V: Send + 'static,
    I: Iterator<Item = PositionMetrics<S, A, V>>,
    Map: Mapper<S, A, V, Te> + Send + Sync + 'static,
{
    info!(
        "Training from {} to {}",
        source_model_info.get_model_name(),
        target_model_info.get_model_name()
    );

    let model_options = get_options(source_model_info)?;
    let moves_left_size = model_options.moves_left_size;
    let source_paths = Paths::from_model_info(&source_model_info);
    let source_base_path = source_paths.get_base_path();

    let mut train_data_file_names = vec![];
    let mut handles = vec![];

    for (i, sample_metrics) in sample_metrics
        .chunks(TRAIN_DATA_CHUNK_SIZE)
        .into_iter()
        .enumerate()
    {
        let train_data_file_name = format!("training_data_{}.npy", i);
        let train_data_path = source_base_path.join(&train_data_file_name);
        info!("Writing data to {:?}", &train_data_path);
        train_data_file_names.push(train_data_file_name);
        let sample_metrics_chunk = sample_metrics.collect::<Vec<_>>();
        let mapper = mapper.clone();

        handles.push(std::thread::spawn(move || {
            let rng = &mut rand::thread_rng();
            let mut wtr = npy::OutFile::open(train_data_path).unwrap();

            for metric in sample_metrics_chunk {
                let metric_symmetires = mapper.get_symmetries(metric);
                let metric = metric_symmetires
                    .choose(rng)
                    .expect("Expected at least one metric to return from symmetries.");

                for record in mapper
                    .game_state_to_input(&metric.game_state, Mode::Train)
                    .into_iter()
                    .map(f16::to_f32)
                    .chain(
                        mapper
                            .policy_metrics_to_expected_output(&metric.game_state, &metric.policy)
                            .into_iter()
                            .chain(
                                std::iter::once(
                                    mapper.map_value_to_value_output(
                                        &metric.game_state,
                                        &metric.score,
                                    ),
                                )
                                .chain(map_moves_left_to_one_hot(
                                    metric.moves_left,
                                    moves_left_size,
                                )),
                            ),
                    )
                {
                    wtr.push(&record).unwrap();
                }
            }

            wtr.close().unwrap();
        }));
    }

    for handle in handles {
        handle
            .join()
            .map_err(|_| anyhow!("Thread failed to write training data"))?;
    }

    let train_data_paths = train_data_file_names.iter().map(|file_name| {
        format!(
            "/{game_name}_runs/{run_name}/{file_name}",
            game_name = source_model_info.get_game_name(),
            run_name = source_model_info.get_run_name(),
            file_name = file_name
        )
    });

    let docker_cmd = format!("docker run --rm \
        --gpus all \
        --mount type=bind,source=\"$(pwd)/{game_name}_runs\",target=/{game_name}_runs \
        -e SOURCE_MODEL_PATH=/{game_name}_runs/{run_name}/models/{game_name}_{run_name}_{source_model_num:0>5}.h5 \
        -e TARGET_MODEL_PATH=/{game_name}_runs/{run_name}/models/{game_name}_{run_name}_{target_model_num:0>5}.h5 \
        -e EXPORT_MODEL_PATH=/{game_name}_runs/{run_name}/exported_models/{target_model_num} \
        -e TENSOR_BOARD_PATH=/{game_name}_runs/{run_name}/tensorboard \
        -e INITIAL_EPOCH={initial_epoch} \
        -e DATA_PATHS={train_data_paths} \
        -e TRAIN_RATIO={train_ratio} \
        -e TRAIN_BATCH_SIZE={train_batch_size} \
        -e EPOCHS={epochs} \
        -e MAX_GRAD_NORM={max_grad_norm} \
        -e LEARNING_RATE={learning_rate} \
        -e POLICY_LOSS_WEIGHT={policy_loss_weight} \
        -e VALUE_LOSS_WEIGHT={value_loss_weight} \
        -e MOVES_LEFT_LOSS_WEIGHT={moves_left_loss_weight} \
        -e INPUT_H={input_h} \
        -e INPUT_W={input_w} \
        -e INPUT_C={input_c} \
        -e OUTPUT_SIZE={output_size} \
        -e MOVES_LEFT_SIZE={moves_left_size} \
        -e NUM_FILTERS={num_filters} \
        -e NUM_BLOCKS={num_blocks} \
        -e NVIDIA_VISIBLE_DEVICES=1 \
        -e CUDA_VISIBLE_DEVICES=1 \
        -e TF_FORCE_GPU_ALLOW_GROWTH=true \
        quoridor_engine/train:latest",
        game_name = source_model_info.get_game_name(),
        run_name = source_model_info.get_run_name(),
        source_model_num = source_model_info.get_model_num(),
        target_model_num = target_model_info.get_model_num(),
        train_ratio = options.train_ratio,
        train_batch_size = options.train_batch_size,
        epochs = (source_model_info.get_model_num() - 1) + options.epochs,
        initial_epoch = (source_model_info.get_model_num() - 1),
        train_data_paths = train_data_paths.map(|p| format!("\"{}\"", p)).join(","),
        learning_rate = options.learning_rate,
        max_grad_norm = options.max_grad_norm,
        policy_loss_weight = options.policy_loss_weight,
        value_loss_weight = options.value_loss_weight,
        moves_left_loss_weight = options.moves_left_loss_weight,
        input_h = model_options.channel_height,
        input_w = model_options.channel_width,
        input_c = model_options.channels,
        output_size = model_options.output_size,
        moves_left_size = model_options.moves_left_size,
        num_filters = model_options.num_filters,
        num_blocks = model_options.num_blocks,
    );

    run_cmd(&docker_cmd)?;

    for file_name in train_data_file_names {
        let path = source_base_path.join(file_name);
        fs::remove_file(path)?;
    }

    create_tensorrt_model(
        source_model_info.get_game_name(),
        source_model_info.get_run_name(),
        target_model_info.get_model_num(),
    )?;

    info!("Training process complete");

    Ok(())
}

#[allow(non_snake_case)]
fn create(model_info: &ModelInfo, options: &TensorflowModelOptions) -> Result<()> {
    let game_name = model_info.get_game_name();
    let run_name = model_info.get_run_name();

    let model_dir = get_model_dir(model_info);
    fs::create_dir_all(model_dir)?;

    write_options(model_info, options)?;

    let docker_cmd = format!(
        "docker run --rm \
        --gpus all \
        --mount type=bind,source=\"$(pwd)/{game_name}_runs\",target=/{game_name}_runs \
        -e TARGET_MODEL_PATH=/{game_name}_runs/{run_name}/models/{game_name}_{run_name}_00001.h5 \
        -e EXPORT_MODEL_PATH=/{game_name}_runs/{run_name}/exported_models/1 \
        -e INPUT_H={input_h} \
        -e INPUT_W={input_w} \
        -e INPUT_C={input_c} \
        -e OUTPUT_SIZE={output_size} \
        -e MOVES_LEFT_SIZE={moves_left_size} \
        -e NUM_FILTERS={num_filters} \
        -e NUM_BLOCKS={num_blocks} \
        -e NVIDIA_VISIBLE_DEVICES=1 \
        -e CUDA_VISIBLE_DEVICES=1 \
        -e TF_FORCE_GPU_ALLOW_GROWTH=true \
        quoridor_engine/create:latest",
        game_name = game_name,
        run_name = run_name,
        input_h = options.channel_height,
        input_w = options.channel_width,
        input_c = options.channels,
        output_size = options.output_size,
        moves_left_size = options.moves_left_size,
        num_filters = options.num_filters,
        num_blocks = options.num_blocks,
    );

    run_cmd(&docker_cmd)?;

    create_tensorrt_model(game_name, run_name, 1)?;

    info!("Model creation process complete");

    Ok(())
}

fn get_model_dir(model_info: &ModelInfo) -> PathBuf {
    Paths::from_model_info(model_info).get_models_path()
}

fn get_model_options_path(model_info: &ModelInfo) -> PathBuf {
    get_model_dir(model_info).join("model-options.json")
}

fn get_options(model_info: &ModelInfo) -> Result<TensorflowModelOptions> {
    let file_path = get_model_options_path(model_info);
    let file_path_lossy = format!("{}", file_path.to_string_lossy());
    let file = File::open(file_path).context(file_path_lossy)?;
    let reader = BufReader::new(file);
    let options = serde_json::from_reader(reader)?;
    Ok(options)
}

fn write_options(model_info: &ModelInfo, options: &TensorflowModelOptions) -> Result<()> {
    let serialized_options = serde_json::to_string(options)?;

    let file_path = get_model_options_path(model_info);
    let file_path_lossy = format!("{}", file_path.to_string_lossy());
    let mut file = File::create(file_path).context(file_path_lossy)?;
    writeln!(file, "{}", serialized_options)?;

    Ok(())
}

fn create_tensorrt_model(game_name: &str, run_name: &str, model_num: usize) -> Result<()> {
    let docker_cmd = format!(
        "docker run --rm \
        --gpus all \
        -e NVIDIA_VISIBLE_DEVICES=1 \
        -e CUDA_VISIBLE_DEVICES=1 \
        -e TF_FORCE_GPU_ALLOW_GROWTH=true \
        --mount type=bind,source=\"$(pwd)/{game_name}_runs\",target=/{game_name}_runs \
        tensorflow/tensorflow:latest-gpu \
        usr/local/bin/saved_model_cli convert \
        --dir /{game_name}_runs/{run_name}/exported_models/{model_num} \
        --output_dir /{game_name}_runs/{run_name}/tensorrt_models/{model_num} \
        --tag_set serve \
        tensorrt \
        --precision_mode FP16 \
        --max_batch_size 512 \
        --is_dynamic_op False",
        game_name = game_name,
        run_name = run_name,
        model_num = model_num
    );

    run_cmd(&docker_cmd)?;

    Ok(())
}

fn run_cmd(cmd: &str) -> Result<()> {
    info!("\n");
    info!("{}", cmd);
    info!("\n");

    let mut cmd = Command::new("/bin/bash")
        .arg("-c")
        .arg(cmd)
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()?;

    let result = cmd.wait();

    info!("OUTPUT: {:?}", result);

    Ok(())
}

struct BatchingModel<S, A, V, E, Map, Te> {
    states_to_analyse: ConcurrentChunkQueue<(usize, S, Waker)>,
    states_analysed: CompletedAnalysisOrdered<A, V>,
    transposition_table: Arc<Option<TranspositionTable<Te>>>,
    engine: E,
    mapper: Arc<Map>,
    reporter: Arc<Reporter<Te>>,
}

impl<S, A, V, E, Map, Te> BatchingModel<S, A, V, E, Map, Te>
where
    S: PartialEq + Hash,
    A: Clone,
    V: Value,
    E: GameEngine<State = S, Action = A, Value = V>,
    Map: Mapper<S, A, V, Te>,
    Te: Send + 'static,
{
    fn new(
        engine: E,
        mapper: Arc<Map>,
        transposition_table: Arc<Option<TranspositionTable<Te>>>,
        reporter: Arc<Reporter<Te>>,
        analysis_request_threads: usize,
        batch_size: usize,
    ) -> Self {
        let analysis_capacity = batch_size * analysis_request_threads;
        let states_to_analyse = ConcurrentChunkQueue::new(batch_size, analysis_capacity);
        let states_analysed = CompletedAnalysisOrdered::with_capacity(analysis_capacity);

        Self {
            states_to_analyse,
            states_analysed,
            transposition_table,
            reporter,
            engine,
            mapper,
        }
    }

    fn collect_analysis<I: Iterator<Item = (usize, S, Waker)>>(
        &self,
        states: I,
        analysis: AnalysisResults,
        output_size: usize,
        moves_left_size: usize,
    ) -> Vec<(usize, GameStateAnalysis<A, V>, Waker)> {
        let mut analysis_len = 0;
        let mut policy_head_iter = analysis.policy_head_output.iter();
        let mut value_head_iter = analysis.value_head_output.iter();
        let mut moves_left_head_iter = analysis
            .moves_left_head_output
            .as_ref()
            .map(|ml| ml.iter().map(|v| v.to_f32()));

        let states_to_wake = states
            .map(|(id, game_state, waker)| {
                let mapper = &*self.mapper;
                let transposition_table_entry = mapper.map_output_to_transposition_entry(
                    &game_state,
                    policy_head_iter.by_ref().take(output_size).copied(),
                    *value_head_iter
                        .next()
                        .expect("Expected value score to exist"),
                    moves_left_head_iter
                        .as_mut()
                        .map(|iter| moves_left_expected_value(iter.by_ref().take(moves_left_size)))
                        .unwrap_or(0.0),
                );
                let analysis = mapper
                    .map_transposition_entry_to_analysis(&game_state, &transposition_table_entry);
                if let Some(transposition_table) = &*self.transposition_table {
                    let transposition_key = self.mapper.get_transposition_key(&game_state);
                    transposition_table.set(transposition_key, transposition_table_entry);
                }

                analysis_len += 1;

                (id, analysis, waker)
            })
            .collect();

        debug_assert!(
            policy_head_iter.next().is_none(),
            "Not all policy head values were used."
        );
        debug_assert!(
            value_head_iter.next().is_none(),
            "Not all value head values were used."
        );
        debug_assert!(
            moves_left_head_iter
                .and_then(|mut iter| iter.next())
                .is_none(),
            "Not all moves left head values were used."
        );

        self.reporter.set_batch_size(analysis_len);

        states_to_wake
    }

    fn try_immediate_analysis(&self, game_state: &S) -> Option<GameStateAnalysis<A, V>> {
        if let Some(value) = self.engine.is_terminal_state(game_state) {
            self.reporter.set_terminal();

            return Some(GameStateAnalysis::new(value, Vec::new(), 0f32));
        }

        if let Some(transposition_table) = &*self.transposition_table {
            if let Some(transposition_entry) =
                transposition_table.get(self.mapper.get_transposition_key(game_state))
            {
                let analysis = self
                    .mapper
                    .map_transposition_entry_to_analysis(game_state, &*transposition_entry);
                drop(transposition_entry);
                self.reporter.set_cache_hit();

                return Some(analysis);
            }

            self.reporter.set_cache_miss();
        }

        self.reporter.set_analyzed_node();

        None
    }

    fn poll(&self, id: usize) -> Poll<GameStateAnalysis<A, V>> {
        let analysis = self.states_analysed.remove(id);

        match analysis {
            Some(analysis) => Poll::Ready(analysis),
            None => Poll::Pending,
        }
    }

    fn request(&self, game_state: S, waker: &Waker) -> usize {
        let id = self.states_analysed.get_id();

        self.states_to_analyse.push((id, game_state, waker.clone()));

        id
    }
}

struct StateToWake {
    id: usize,
    waker: Waker,
}

impl Ord for StateToWake {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.id.cmp(&self.id)
    }
}

impl PartialOrd for StateToWake {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(other.id.cmp(&self.id))
    }
}

impl PartialEq for StateToWake {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for StateToWake {}

struct CompletedAnalysisOrdered<A, V> {
    states_analysed: IncrementingMap<GameStateAnalysis<A, V>>,
    states_to_wake: Mutex<(usize, BinaryHeap<StateToWake>)>,
    id_generator: AtomicUsize,
}

impl<A, V> CompletedAnalysisOrdered<A, V> {
    fn with_capacity(n: usize) -> Self {
        Self {
            states_analysed: IncrementingMap::with_capacity(n),
            states_to_wake: Mutex::new((1, BinaryHeap::with_capacity(n))),
            id_generator: AtomicUsize::new(1),
        }
    }

    fn get_id(&self) -> usize {
        self.id_generator.fetch_add(1, Ordering::Relaxed)
    }

    fn wake_with_analysis<I>(&self, analysis: I)
    where
        I: Iterator<Item = (usize, GameStateAnalysis<A, V>, Waker)>,
    {
        let mut lock = self.states_to_wake.lock();
        let (ref mut next_id_to_wake, ref mut states_to_wake) = *lock;

        for (id, analysis, waker) in analysis {
            self.states_analysed.insert(id, analysis);

            if id == *next_id_to_wake {
                *next_id_to_wake += 1;
                waker.wake();

                while let Some(val) = states_to_wake.peek_mut() {
                    if val.id == *next_id_to_wake {
                        let state_to_wake = PeekMut::pop(val);
                        *next_id_to_wake += 1;
                        state_to_wake.waker.wake();
                    } else {
                        break;
                    }
                }
            } else {
                states_to_wake.push(StateToWake { id, waker })
            }
        }
    }

    fn remove(&self, id: usize) -> Option<GameStateAnalysis<A, V>> {
        self.states_analysed.remove(id)
    }
}

pub struct GameStateAnalysisFuture<S, A, V, E, Map, Te> {
    game_state: Option<S>,
    id: usize,
    batching_model: Arc<BatchingModel<S, A, V, E, Map, Te>>,
}

impl<S, A, V, E, Map, Te> GameStateAnalysisFuture<S, A, V, E, Map, Te> {
    fn new(game_state: S, batching_model: Arc<BatchingModel<S, A, V, E, Map, Te>>) -> Self {
        Self {
            id: 0,
            game_state: Some(game_state),
            batching_model,
        }
    }
}

impl<S, A, V, E, Map, Te> Future for GameStateAnalysisFuture<S, A, V, E, Map, Te>
where
    S: PartialEq + Hash + Unpin,
    A: Clone,
    V: Value,
    E: GameEngine<State = S, Action = A, Value = V> + Send + Sync + 'static,
    Map: Mapper<S, A, V, Te>,
    Te: Send + 'static,
{
    type Output = GameStateAnalysis<A, V>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if self.id == 0 {
            let self_mut = self.get_mut();
            let game_state = self_mut
                .game_state
                .take()
                .expect("Expected game_state to exist");

            self_mut.id = self_mut.batching_model.request(game_state, cx.waker());

            Poll::Pending
        } else {
            (*self).batching_model.poll(self.id)
        }
    }
}

pub fn moves_left_expected_value<I: Iterator<Item = f32>>(moves_left_scores: I) -> f32 {
    moves_left_scores
        .enumerate()
        .map(|(i, s)| (i + 1) as f32 * s)
        .fold(0.0f32, |s, e| s + e)
}

fn map_moves_left_to_one_hot(moves_left: usize, moves_left_size: usize) -> Vec<f32> {
    if moves_left_size == 0 {
        return vec![];
    }

    let moves_left = moves_left.max(0).min(moves_left_size);
    let mut moves_left_one_hot = vec![0f32; moves_left_size];
    moves_left_one_hot[moves_left - 1] = 1.0;

    moves_left_one_hot
}

fn fill_tensor<I: Iterator<Item = T>, T: tensorflow::TensorType>(tensor: &mut Tensor<T>, iter: I) {
    for (e, v) in tensor.iter_mut().zip(iter) {
        e.clone_from(&v);
    }
}
