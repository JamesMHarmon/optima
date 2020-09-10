use anyhow::{anyhow, ensure, Context as AnyhowContext, Result};
use half::f16;
use itertools::Itertools;
use log::info;
use parking_lot::Mutex;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use std::cell::Cell;
use std::collections::binary_heap::PeekMut;
use std::collections::BinaryHeap;
use std::fs::{self, File};
use std::future::Future;
use std::hash::Hash;
use std::io::{BufReader, Write};
use std::path::PathBuf;
use std::pin::Pin;
use std::process::{Command, Stdio};
use std::rc::Rc;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use std::task::{Context, Poll, Waker};
use std::time::Instant;
use tensorflow::{Graph, Operation, Session, SessionOptions, SessionRunArgs, Tensor};

use common::chunk_queue::ConcurrentChunkQueue;
use common::incrementing_map::IncrementingMap;
use engine::engine::GameEngine;
use engine::game_state::GameState;
use engine::value::Value;

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
        let batching_model = Arc::new(BatchingModel::new(
            engine,
            mapper.clone(),
            transposition_table,
            analysis_request_threads,
            batch_size,
        ));
        let options = get_options(&model_info).expect("Could not load model options file");

        Self {
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
    ) {
        loop {
            let thread_num = active_threads.fetch_add(1, Ordering::Relaxed);

            if thread_num >= analysis_request_threads {
                active_threads.fetch_sub(1, Ordering::Relaxed);
                break;
            }

            let batching_model_ref = batching_model.clone();
            let model_info = model_info.clone();
            let mapper = mapper.clone();
            let active_analyzers = active_analyzers.clone();
            let active_threads = active_threads.clone();

            std::thread::spawn(move || {
                let mut last_report = Instant::now();
                let input_dim = mapper.get_input_dimensions();
                let predictor = Predictor::new(&model_info, input_dim);

                loop {
                    let (states_to_analyse, analyzed_states) =
                        (&(*batching_model_ref).states_to_analyse).dequeue_chunk_with_split(
                            |(id, state_to_analyse, waker)| match Self::try_immediate_analysis(
                                &*batching_model_ref,
                                &state_to_analyse,
                            ) {
                                Some(analysis) => Err((id, analysis, waker)),
                                None => Ok((id, state_to_analyse, waker)),
                            },
                        );

                    if states_to_analyse.is_empty() && analyzed_states.is_empty() {
                        if active_analyzers.load(Ordering::Relaxed) == 0 {
                            break;
                        }

                        std::thread::sleep(std::time::Duration::from_millis(1));
                    }

                    if !states_to_analyse.is_empty() {
                        let game_states_to_predict =
                            states_to_analyse.iter().map(|(_, game_state, _)| {
                                mapper.game_state_to_input(game_state, Mode::Infer)
                            });
                        let predictions = predictor
                            .predict(game_states_to_predict, states_to_analyse.len())
                            .expect("Expected predict to be successful");
                        batching_model_ref
                            .provide_analysis(
                                states_to_analyse.into_iter(),
                                predictions,
                                output_size,
                                moves_left_size,
                            )
                            .expect("Expected provide_analysis to be successful");
                    }

                    for (id, analysis, waker) in analyzed_states {
                        batching_model_ref
                            .states_analysed
                            .wake_with_analysis(id, analysis, waker);
                    }

                    if thread_num == 0 {
                        let elapsed_mills = last_report.elapsed().as_millis();
                        if elapsed_mills >= 5_000 {
                            let transposition_hits = batching_model_ref.take_transposition_hits();
                            let num_infer_nodes = batching_model_ref.take_num_nodes_analysed();
                            let num_transpo_nodes = transposition_hits
                                .map_or(0, |(_entries, _capacity, hits, _misses)| hits);
                            let (min_batch_size, max_batch_size) =
                                batching_model_ref.take_min_max_batch_size();
                            let infer_nps = num_infer_nodes as f32 * 1000.0 / elapsed_mills as f32;
                            let total_nps = (num_infer_nodes + num_transpo_nodes) as f32 * 1000.0
                                / elapsed_mills as f32;
                            info!(
                                "NPS: {total_nps:.2}, Infered NPS: {infer_nps:.2}, Min Batch Size: {min_batch_size}, Max Batch Size: {max_batch_size}",
                                total_nps=total_nps,
                                infer_nps=infer_nps,
                                min_batch_size=min_batch_size,
                                max_batch_size=max_batch_size
                            );
                            if let Some((entries, capacity, hits, misses)) = transposition_hits {
                                info!(
                                    "Hits: %{:.2}, Cache Hydration: %{:.2}, Entries: {}, Capacity: {}",
                                    if hits > 0 {
                                        (hits as f32 / (hits + misses) as f32) * 100f32
                                    } else {
                                        0f32
                                    },
                                    (entries as f32 / capacity as f32) * 100f32,
                                    entries,
                                    capacity
                                );
                            }
                            last_report = Instant::now();
                        }
                    }
                }

                active_threads.fetch_sub(1, Ordering::Relaxed);
            });
        }
    }

    fn try_immediate_analysis(
        batching_model: &BatchingModel<S, A, V, E, Map, Te>,
        game_state: &S,
    ) -> Option<GameStateAnalysis<A, V>> {
        if let Some(value) = batching_model.engine.is_terminal_state(game_state) {
            batching_model
                .num_nodes_analysed
                .fetch_add(1, Ordering::Relaxed);

            return Some(GameStateAnalysis::new(value, Vec::new(), 0f32));
        }

        if let Some(transposition_table) = &*batching_model.transposition_table {
            if let Some(transposition_entry) =
                transposition_table.get(batching_model.mapper.get_transposition_key(game_state))
            {
                let analysis = batching_model
                    .mapper
                    .map_transposition_entry_to_analysis(game_state, &*transposition_entry);
                drop(transposition_entry);
                batching_model.cache_hits.fetch_add(1, Ordering::Relaxed);

                return Some(analysis);
            }
        }

        batching_model.cache_misses.fetch_add(1, Ordering::Relaxed);

        None
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
    num_nodes_analysed: AtomicUsize,
    min_batch_size: AtomicUsize,
    max_batch_size: AtomicUsize,
    cache_misses: AtomicUsize,
    cache_hits: AtomicUsize,
    engine: E,
    mapper: Arc<Map>,
}

impl<S, A, V, E, Map, Te> BatchingModel<S, A, V, E, Map, Te>
where
    S: PartialEq + Hash,
    A: Clone,
    V: Value,
    E: GameEngine<State = S, Action = A, Value = V>,
    Map: Mapper<S, A, V, Te>,
{
    fn new(
        engine: E,
        mapper: Arc<Map>,
        transposition_table: Arc<Option<TranspositionTable<Te>>>,
        analysis_request_threads: usize,
        batch_size: usize,
    ) -> Self {
        let analysis_capacity = batch_size * analysis_request_threads;
        let states_to_analyse = ConcurrentChunkQueue::new(batch_size, analysis_capacity);
        let states_analysed = CompletedAnalysisOrdered::with_capacity(analysis_capacity);
        let num_nodes_analysed = AtomicUsize::new(0);
        let min_batch_size = AtomicUsize::new(std::usize::MAX);
        let max_batch_size = AtomicUsize::new(0);
        let cache_misses = AtomicUsize::new(0);
        let cache_hits = AtomicUsize::new(0);

        Self {
            states_to_analyse,
            states_analysed,
            transposition_table,
            num_nodes_analysed,
            min_batch_size,
            max_batch_size,
            cache_misses,
            cache_hits,
            engine,
            mapper,
        }
    }

    fn provide_analysis<I: Iterator<Item = (usize, S, Waker)>>(
        &self,
        states: I,
        analysis: AnalysisResults,
        output_size: usize,
        moves_left_size: usize,
    ) -> Result<()> {
        let mut analysis_len = 0;
        let mut policy_head_iter = analysis.policy_head_output.iter();
        let mut value_head_iter = analysis.value_head_output.iter();
        let mut moves_left_head_iter = analysis
            .moves_left_head_output
            .as_ref()
            .map(|ml| ml.iter().map(|v| v.to_f32()));

        for (id, game_state, waker) in states {
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

            let analysis =
                mapper.map_transposition_entry_to_analysis(&game_state, &transposition_table_entry);

            if let Some(transposition_table) = &*self.transposition_table {
                let transposition_key = self.mapper.get_transposition_key(&game_state);
                transposition_table.set(transposition_key, transposition_table_entry);
            }

            self.states_analysed.wake_with_analysis(id, analysis, waker);

            analysis_len += 1;
        }

        ensure!(
            policy_head_iter.next().is_none(),
            "Not all policy head values were used."
        );
        ensure!(
            value_head_iter.next().is_none(),
            "Not all value head values were used."
        );
        ensure!(
            moves_left_head_iter
                .and_then(|mut iter| iter.next())
                .is_none(),
            "Not all moves left head values were used."
        );

        self.min_batch_size
            .fetch_min(analysis_len, Ordering::Relaxed);
        self.max_batch_size
            .fetch_max(analysis_len, Ordering::Relaxed);

        Ok(())
    }

    fn take_num_nodes_analysed(&self) -> usize {
        self.num_nodes_analysed.swap(0, Ordering::Relaxed)
    }

    fn take_min_max_batch_size(&self) -> (usize, usize) {
        (
            self.min_batch_size.swap(std::usize::MAX, Ordering::Relaxed),
            self.max_batch_size.swap(0, Ordering::Relaxed),
        )
    }

    fn take_transposition_hits(&self) -> Option<(usize, usize, usize, usize)> {
        self.transposition_table
            .as_ref()
            .as_ref()
            .map(|transposition_table| {
                (
                    transposition_table.num_entries(),
                    transposition_table.capacity(),
                    self.cache_hits.swap(0, Ordering::Relaxed),
                    self.cache_misses.swap(0, Ordering::Relaxed),
                )
            })
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

// struct StateToWake {
//     id: usize,
//     waker: Waker,
// }

struct CompletedAnalysisOrdered<A, V> {
    states_analysed: IncrementingMap<GameStateAnalysis<A, V>>,
    // states_to_wake: Mutex<(usize, BinaryHeap<StateToWake>)>,
    id_generator: AtomicUsize,
}

impl<A, V> CompletedAnalysisOrdered<A, V> {
    fn with_capacity(n: usize) -> Self {
        Self {
            states_analysed: IncrementingMap::with_capacity(n),
            // states_to_wake: Mutex::new((1, BinaryHeap::with_capacity(n))),
            id_generator: AtomicUsize::new(1),
        }
    }

    fn get_id(&self) -> usize {
        self.id_generator.fetch_add(1, Ordering::Relaxed)
    }

    fn wake_with_analysis(&self, id: usize, analysis: GameStateAnalysis<A, V>, waker: Waker) {
        self.states_analysed.insert(id, analysis);

        waker.wake();

        // let lock = self.states_to_wake.lock();
        // let (ref next_id_to_wake, states_to_wake) = *lock;

        // if id == *next_id_to_wake {
        //     *next_id_to_wake += 1;
        //     waker.wake();

        //     loop {
        //         if let Some(val) = states_to_wake.peek_mut() {
        //             let StateToWake { id, waker } = *val;
        //             if id == *next_id_to_wake {
        //                 let state_to_wake = PeekMut::pop(val);
        //                 *next_id_to_wake += 1;
        //                 state_to_wake.waker.wake();
        //             }
        //         }
        //     }
        // } else {
        //     states_to_wake.push(StateToWake { id, waker })
        // }
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
