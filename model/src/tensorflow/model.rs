use anyhow::{anyhow, Context as AnyhowContext, Result};
use crossbeam::{Receiver as UnboundedReceiver, Sender as UnboundedSender};
use engine::engine::GameEngine;
use engine::game_state::GameState;
use engine::value::Value;
use half::f16;
use itertools::Itertools;
use log::info;
use parking_lot::Mutex;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
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
use std::task::{Context, Poll};
use std::{collections::binary_heap::PeekMut, sync::Weak};
use tensorflow::{Graph, Operation, Session, SessionOptions, SessionRunArgs, Tensor, TensorType};
use tokio::sync::{oneshot, oneshot::Receiver, oneshot::Sender};

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
    batching_model: Mutex<Weak<BatchingModelAndSender<S, A, V, E, Map, Te>>>,
    analysis_request_threads: usize,
    engine: Arc<E>,
    mapper: Arc<Map>,
    options: TensorflowModelOptions,
    batch_size: usize,
    tt_cache_size: usize,
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
        let batch_size = std::env::var("ANALYSIS_REQUEST_BATCH_SIZE")
            .map(|v| {
                v.parse::<usize>()
                    .expect("ANALYSIS_REQUEST_BATCH_SIZE must be a valid int")
            })
            .unwrap_or(ANALYSIS_REQUEST_BATCH_SIZE);

        let analysis_request_threads = std::env::var("ANALYSIS_REQUEST_THREADS")
            .map(|v| {
                v.parse::<usize>()
                    .expect("ANALYSIS_REQUEST_THREADS must be a valid int")
            })
            .unwrap_or(ANALYSIS_REQUEST_THREADS);

        if std::env::var("TF_CPP_MIN_LOG_LEVEL").is_err() {
            std::env::set_var("TF_CPP_MIN_LOG_LEVEL", "2");
        }

        let options = get_options(&model_info).expect("Could not load model options file");

        let mapper = Arc::new(mapper);
        let engine = Arc::new(engine);
        let batching_model = Mutex::new(Weak::new());

        Self {
            analysis_request_threads,
            batch_size,
            model_info,
            batching_model,
            engine,
            mapper,
            tt_cache_size,
            options,
        }
    }

    pub fn create(model_info: &ModelInfo, options: &TensorflowModelOptions) -> Result<()> {
        create(model_info, options)
    }

    fn create_batching_model(
        &self,
        receiver: UnboundedReceiver<StatesToAnalyse<S, A, V>>,
    ) -> BatchingModel<E, Map, Te> {
        let transposition_table = Arc::new(if self.tt_cache_size > 0 {
            Some(TranspositionTable::new(self.tt_cache_size))
        } else {
            None
        });

        let reporter = Arc::new(Reporter::new(transposition_table.clone()));

        BatchingModel::new(
            self.engine.clone(),
            self.mapper.clone(),
            transposition_table,
            reporter,
            self.model_info.clone(),
            self.batch_size,
            self.analysis_request_threads,
            self.options.output_size,
            self.options.moves_left_size,
            receiver,
        )
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
        let batching_model_ref = &mut *self.batching_model.lock();
        let batching_model = batching_model_ref.upgrade();

        let batching_model = match batching_model {
            None => {
                let (sender, receiver) = crossbeam::unbounded();
                let batching_model_arc = Arc::new((self.create_batching_model(receiver), sender));

                *batching_model_ref = Arc::downgrade(&batching_model_arc);

                batching_model_arc
            }
            Some(model_sender) => model_sender,
        };

        Self::Analyzer::new(batching_model)
    }
}

struct Predictor {
    session: SessionAndOps,
}

impl Predictor {
    fn new(model_info: &ModelInfo) -> Self {
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
            session: SessionAndOps {
                session,
                op_input,
                op_value_head,
                op_policy_head,
                op_moves_left_head,
            },
        }
    }

    fn predict(&self, tensor: &Tensor<f16>) -> Result<AnalysisResults> {
        let session = &self.session;

        let mut output_step = SessionRunArgs::new();
        output_step.add_feed(&session.op_input, 0, &tensor);
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

    fn fill_tensor<'a, T: Iterator<Item = &'a [f16]>>(tensor: &mut Tensor<f16>, inputs: T) {
        for (i, input) in inputs.enumerate() {
            let input_width = input.len();
            tensor[(input_width * i)..(input_width * (i + 1))].copy_from_slice(&input);
        }
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

type BatchingModelAndSender<S, A, V, E, Map, Te> = (
    BatchingModel<E, Map, Te>,
    UnboundedSender<StatesToAnalyse<S, A, V>>,
);

pub struct GameAnalyzer<S, A, V, E, Map, Te> {
    batching_model: Arc<BatchingModelAndSender<S, A, V, E, Map, Te>>,
    analysed_state_ordered: CompletedAnalysisOrdered,
    analysed_state_sender: UnboundedSender<AnalysisToSend<A, V>>,
}

impl<S, A, V, E, Map, Te> GameAnalyzer<S, A, V, E, Map, Te>
where
    A: Send + 'static,
    V: Send + 'static,
{
    fn new(batching_model: Arc<BatchingModelAndSender<S, A, V, E, Map, Te>>) -> Self {
        // @TODO: Hardcoded
        let (analysed_state_sender, analyzed_state_receiver) = crossbeam::channel::unbounded();

        let analysed_state_ordered =
            CompletedAnalysisOrdered::with_capacity(analyzed_state_receiver, 1);

        Self {
            batching_model,
            analysed_state_ordered,
            analysed_state_sender,
        }
    }
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
    type Future = UnwrappedReceiver<GameStateAnalysis<A, V>>;

    /// Outputs a value from [-1, 1] depending on the player to move's evaluation of the current state.
    /// If the evaluation is a draw then 0.0 will be returned.
    /// Along with the value output a list of policy scores for all VALID moves is returned. If the position
    /// is terminal then the vector will be empty.
    fn get_state_analysis(&self, game_state: &S) -> UnwrappedReceiver<GameStateAnalysis<A, V>> {
        let (tx, rx) = oneshot::channel();
        let id = self.analysed_state_ordered.get_id();
        let sender = &self.batching_model.1;
        sender
            .send((
                id,
                game_state.to_owned(),
                self.analysed_state_sender.clone(),
                tx,
            ))
            .unwrap_or_else(|_| panic!("Channel Failure 3"));

        UnwrappedReceiver::new(rx)
    }
}

use pin_project::pin_project;

#[pin_project]
pub struct UnwrappedReceiver<T> {
    #[pin]
    receiver: Receiver<T>,
}

impl<T> UnwrappedReceiver<T> {
    fn new(receiver: Receiver<T>) -> Self {
        UnwrappedReceiver { receiver }
    }
}

impl<T> Future for UnwrappedReceiver<T> {
    type Output = T;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<T> {
        match self.as_mut().project().receiver.poll(cx) {
            Poll::Ready(val) => Poll::Ready(val.expect("Expected a receivable value")),
            Poll::Pending => Poll::Pending,
        }
    }
}

type StatesToAnalyse<S, A, V> = (
    usize,
    S,
    UnboundedSender<AnalysisToSend<A, V>>,
    Sender<GameStateAnalysis<A, V>>,
);

type AnalysisToSend<A, V> = (
    usize,
    GameStateAnalysis<A, V>,
    Sender<GameStateAnalysis<A, V>>,
);

struct BatchingModel<E, Map, Te> {
    _transposition_table: Arc<Option<TranspositionTable<Te>>>,
    _engine: Arc<E>,
    _mapper: Arc<Map>,
    _reporter: Arc<Reporter<Te>>,
}

impl<S, A, V, E, Map, Te> BatchingModel<E, Map, Te>
where
    S: PartialEq + Hash + Send + 'static,
    A: Clone + Send + 'static,
    V: Value + Send + 'static,
    E: GameEngine<State = S, Action = A, Value = V> + Send + Sync + 'static,
    Map: Mapper<S, A, V, Te> + Send + Sync + 'static,
    Te: Send + 'static,
{
    fn new(
        engine: Arc<E>,
        mapper: Arc<Map>,
        transposition_table: Arc<Option<TranspositionTable<Te>>>,
        reporter: Arc<Reporter<Te>>,
        model_info: ModelInfo,
        batch_size: usize,
        analysis_request_threads: usize,
        output_size: usize,
        moves_left_size: usize,
        states_to_analyse_receiver: UnboundedReceiver<StatesToAnalyse<S, A, V>>,
    ) -> Self {
        Self::create_analysis_tasks(
            engine.clone(),
            mapper.clone(),
            transposition_table.clone(),
            reporter.clone(),
            states_to_analyse_receiver,
            model_info,
            output_size,
            moves_left_size,
            batch_size,
            analysis_request_threads,
        );

        Self {
            _transposition_table: transposition_table,
            _reporter: reporter,
            _engine: engine,
            _mapper: mapper,
        }
    }

    fn create_analysis_tasks(
        engine: Arc<E>,
        mapper: Arc<Map>,
        transposition_table: Arc<Option<TranspositionTable<Te>>>,
        reporter: Arc<Reporter<Te>>,
        receiver: UnboundedReceiver<StatesToAnalyse<S, A, V>>,
        model_info: ModelInfo,
        output_size: usize,
        moves_left_size: usize,
        batch_size: usize,
        analysis_request_threads: usize,
    ) {
        let model_info = model_info.clone();
        let mapper = mapper.clone();

        let (states_to_predict_tx, states_to_predict_rx) = crossbeam::channel::unbounded();

        let transposition_table_clone = transposition_table.clone();
        let mapper_clone = mapper.clone();
        let reporter_clone = reporter.clone();
        tokio::task::spawn_blocking(move || {
            while let Ok((id, state_to_analyse, unordered_tx, tx)) = receiver.recv() {
                match Self::try_immediate_analysis(
                    &state_to_analyse,
                    &*transposition_table_clone,
                    &*engine,
                    &*mapper_clone,
                    &*reporter_clone,
                ) {
                    Some(analysis) => {
                        unordered_tx
                            .send((id, analysis, tx))
                            .unwrap_or_else(|_| panic!("Channel Failure 1"));
                    }
                    None => {
                        let input =
                            mapper_clone.game_state_to_input(&state_to_analyse, Mode::Infer);
                        states_to_predict_tx
                            .send((id, state_to_analyse, input, unordered_tx, tx))
                            .unwrap_or_else(|_| panic!("Channel Failure 2"));
                    }
                };
            }
        });

        let predictor = Arc::new(Predictor::new(&model_info));
        let (predicted_states_tx, predicted_states_rx) = crossbeam::channel::unbounded();

        for _ in 0..analysis_request_threads {
            let mapper = mapper.clone();
            let reporter = reporter.clone();
            let predictor = predictor.clone();
            let states_to_predict_rx = states_to_predict_rx.clone();
            let predicted_states_tx = predicted_states_tx.clone();

            tokio::task::spawn_blocking(move || {
                let dimensions = mapper.get_input_dimensions();
                let mut tensor_pool = TensorPool::new(dimensions);

                while let Ok(state_to_analyze) = states_to_predict_rx.recv() {
                    let mut states_to_analyse = Vec::with_capacity(batch_size);

                    states_to_analyse.push(state_to_analyze);

                    while let Ok(state_to_analyze) = states_to_predict_rx.try_recv() {
                        states_to_analyse.push(state_to_analyze);

                        if states_to_analyse.len() == batch_size {
                            break;
                        }
                    }

                    let tensor = tensor_pool.get(states_to_analyse.len());

                    reporter.set_batch_size(states_to_analyse.len());

                    Predictor::fill_tensor(
                        tensor,
                        states_to_analyse
                            .iter()
                            .map(|(_, _, input, _, _)| input.as_slice()),
                    );

                    let predictions = predictor
                        .predict(tensor)
                        .expect("Expected predict to be successful");

                    predicted_states_tx
                        .send((states_to_analyse, predictions))
                        .unwrap_or_else(|_| panic!("Failed to send value in channel."));
                }
            });
        }

        tokio::task::spawn_blocking(move || {
            while let Ok((analyzed_states, predictions)) = predicted_states_rx.recv() {
                let mut policy_head_iter = predictions.policy_head_output.iter();
                let mut value_head_iter = predictions.value_head_output.iter();
                let mut moves_left_head_iter = predictions
                    .moves_left_head_output
                    .as_ref()
                    .map(|ml| ml.iter().map(|v| v.to_f32()));

                let transposition_table = &*transposition_table;
                for (id, game_state, _, tx, tx2) in analyzed_states.into_iter() {
                    let transposition_table_entry = mapper.map_output_to_transposition_entry(
                        &game_state,
                        policy_head_iter.by_ref().take(output_size).copied(),
                        *value_head_iter
                            .next()
                            .expect("Expected value score to exist"),
                        moves_left_head_iter
                            .as_mut()
                            .map(|iter| {
                                moves_left_expected_value(iter.by_ref().take(moves_left_size))
                            })
                            .unwrap_or(0.0),
                    );

                    let analysis = mapper.map_transposition_entry_to_analysis(
                        &game_state,
                        &transposition_table_entry,
                    );

                    if let Some(transposition_table) = transposition_table {
                        let transposition_key = mapper.get_transposition_key(&game_state);
                        transposition_table.set(transposition_key, transposition_table_entry);
                    }

                    tx.send((id, analysis, tx2))
                        .unwrap_or_else(|_| panic!("Channel Failure 4"));
                }
            }
        });
    }

    fn try_immediate_analysis(
        game_state: &S,
        transposition_table: &Option<TranspositionTable<Te>>,
        engine: &E,
        mapper: &Map,
        reporter: &Reporter<Te>,
    ) -> Option<GameStateAnalysis<A, V>> {
        if let Some(value) = engine.is_terminal_state(game_state) {
            reporter.set_terminal();

            return Some(GameStateAnalysis::new(value, Vec::new(), 0f32));
        }

        if let Some(transposition_table) = &*transposition_table {
            if let Some(transposition_entry) =
                transposition_table.get(mapper.get_transposition_key(game_state))
            {
                let analysis =
                    mapper.map_transposition_entry_to_analysis(game_state, &*transposition_entry);
                drop(transposition_entry);
                reporter.set_cache_hit();

                return Some(analysis);
            }

            reporter.set_cache_miss();
        }

        reporter.set_analyzed_node();

        None
    }
}

struct StateToTransmit<A, V> {
    id: usize,
    tx: Sender<GameStateAnalysis<A, V>>,
    analysis: GameStateAnalysis<A, V>,
}

impl<A, V> Ord for StateToTransmit<A, V> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.id.cmp(&self.id)
    }
}

impl<A, V> PartialOrd for StateToTransmit<A, V> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(other.id.cmp(&self.id))
    }
}

impl<A, V> PartialEq for StateToTransmit<A, V> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl<A, V> Eq for StateToTransmit<A, V> {}

struct CompletedAnalysisOrdered {
    id_generator: AtomicUsize,
}

impl CompletedAnalysisOrdered {
    fn with_capacity<A, V>(
        analyzed_state_receiver: UnboundedReceiver<AnalysisToSend<A, V>>,
        n: usize,
    ) -> Self
    where
        A: Send + 'static,
        V: Send + 'static,
    {
        Self::create_ordering_task(analyzed_state_receiver, n);

        Self {
            id_generator: AtomicUsize::new(1),
        }
    }

    fn get_id(&self) -> usize {
        self.id_generator.fetch_add(1, Ordering::Relaxed)
    }

    fn create_ordering_task<A, V>(
        receiver: UnboundedReceiver<(
            usize,
            GameStateAnalysis<A, V>,
            Sender<GameStateAnalysis<A, V>>,
        )>,
        capacity: usize,
    ) where
        A: Send + 'static,
        V: Send + 'static,
    {
        tokio::task::spawn_blocking(move || {
            let mut analyzed_states_to_tx =
                BinaryHeap::<StateToTransmit<A, V>>::with_capacity(capacity);
            let mut next_id_to_tx: usize = 1;

            while let Ok(analysed_state) = receiver.recv() {
                let (id, analysis, tx) = analysed_state;
                if id == next_id_to_tx {
                    next_id_to_tx += 1;
                    if tx.send(analysis).is_err() {
                        panic!("Failed to send analysis");
                    }

                    while let Some(val) = analyzed_states_to_tx.peek_mut() {
                        if val.id == next_id_to_tx {
                            let state_to_tx = PeekMut::pop(val);
                            next_id_to_tx += 1;
                            if state_to_tx.tx.send(state_to_tx.analysis).is_err() {
                                panic!("Failed to send analysis");
                            }
                        } else {
                            break;
                        }
                    }
                } else {
                    analyzed_states_to_tx.push(StateToTransmit { id, analysis, tx })
                }
            }
        });
    }
}

struct TensorPool<T: TensorType> {
    tensors: Vec<Tensor<T>>,
    dimensions: [u64; 3],
}

impl<T: TensorType> TensorPool<T> {
    fn new(dimensions: [u64; 3]) -> Self {
        Self {
            tensors: vec![],
            dimensions,
        }
    }

    fn get(&mut self, size: usize) -> &mut Tensor<T> {
        let next_matching_power = (size as f64).log2().ceil() as usize + 1;
        let tensors = &mut self.tensors;
        while tensors.len() < next_matching_power {
            tensors.push(Tensor::new(&[
                2u32.pow(tensors.len() as u32) as u64,
                self.dimensions[0],
                self.dimensions[1],
                self.dimensions[2],
            ]));
        }

        &mut tensors[next_matching_power - 1]
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
