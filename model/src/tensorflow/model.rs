use anyhow::Result;
use crossbeam::channel::{Receiver as UnboundedReceiver, Sender as UnboundedSender};
use engine::engine::GameEngine;
use engine::game_state::GameState;
use engine::value::Value;
use half::f16;
use log::{debug, info};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::BinaryHeap;
use std::future::Future;
use std::hash::Hash;
use std::os::raw::c_int;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};
use std::{collections::binary_heap::PeekMut, sync::Weak};
use tensorflow::*;
use tokio::sync::{mpsc, oneshot, oneshot::Receiver, oneshot::Sender};

use super::super::analytics::{self, GameStateAnalysis};
use super::super::model::{Model as ModelTrait, TrainOptions};
use super::super::model_info::ModelInfo;
use super::super::node_metrics::NodeMetrics;
use super::super::position_metrics::PositionMetrics;
use super::*;

#[cfg_attr(feature = "tensorflow_system_alloc", global_allocator)]
#[cfg(feature = "tensorflow_system_alloc")]
static ALLOCATOR: std::alloc::System = std::alloc::System;

#[allow(clippy::type_complexity)]
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
    S: Hash + Send + Sync + 'static,
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
        super::create(model_info, options)
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
        super::train(
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
                let (sender, receiver) = crossbeam::channel::unbounded();
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
            "{game_name}_runs/{run_name}/exported_models/{model_num}",
            game_name = model_info.get_game_name(),
            run_name = model_info.get_run_name(),
            model_num = model_info.get_model_num(),
        );

        let exported_model_path = std::env::current_dir()
            .expect("Expected to be able to open the model directory")
            .join(exported_model_path);

        info!("{:?}", exported_model_path);

        let model = SavedModelBundle::load(
            &SessionOptions::new(),
            &["serve"],
            &mut graph,
            exported_model_path,
        )
        .expect("Expected to be able to load model");

        let signature = model
            .meta_graph_def()
            .get_signature(DEFAULT_SERVING_SIGNATURE_DEF_KEY)
            .unwrap_or_else(|_| {
                panic!(
                    "Failed to get signature: {}",
                    DEFAULT_SERVING_SIGNATURE_DEF_KEY
                )
            });

        let input_tensor_info = signature
            .inputs()
            .iter()
            .next()
            .expect("Expect at least one input in signature")
            .1;
        let value_head_tensor_info = signature
            .outputs()
            .iter()
            .find(|(s, _)| s.contains("value_head"))
            .expect("'value_head' output in signature not found")
            .1;
        let policy_head_tensor_info = signature
            .outputs()
            .iter()
            .find(|(s, _)| s.contains("policy_head"))
            .expect("'policy_head' output in signature not found")
            .1;
        let moves_left_head_tensor_info = signature
            .outputs()
            .iter()
            .find(|(s, _)| s.contains("moves_left_head"))
            .map(|(_, s)| s);

        let input_name = &input_tensor_info.name().name;
        let value_head_name = &value_head_tensor_info.name().name;
        let policy_head_name = &policy_head_tensor_info.name().name;
        let moves_left_head_name = moves_left_head_tensor_info.map(|x| &x.name().name);

        let op_input = graph
            .operation_by_name_required(input_name)
            .map(|operation| OperationWithIndex {
                operation,
                index: input_tensor_info.name().index,
            })
            .expect("Expected to find input operation");
        let op_value_head = graph
            .operation_by_name_required(value_head_name)
            .map(|operation| OperationWithIndex {
                operation,
                index: value_head_tensor_info.name().index,
            })
            .expect("Expected to find value_head operation");
        let op_policy_head = graph
            .operation_by_name_required(policy_head_name)
            .map(|operation| OperationWithIndex {
                operation,
                index: policy_head_tensor_info.name().index,
            })
            .expect("Expected to find policy_head operation");
        let op_moves_left_head = moves_left_head_name
            .map(|name| {
                graph
                    .operation_by_name_required(name)
                    .map(|operation| OperationWithIndex {
                        operation,
                        index: moves_left_head_tensor_info.unwrap().name().index,
                    })
                    .ok()
            })
            .flatten();

        let session = model.session;

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
        output_step.add_feed(&session.op_input.operation, session.op_input.index, tensor);
        let value_head_fetch_token = output_step.request_fetch(
            &session.op_value_head.operation,
            session.op_value_head.index,
        );
        let policy_head_fetch_token = output_step.request_fetch(
            &session.op_policy_head.operation,
            session.op_policy_head.index,
        );
        let moves_left_head_fetch_token = session
            .op_moves_left_head
            .as_ref()
            .map(|op| output_step.request_fetch(&op.operation, op.index));

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
            tensor[(input_width * i)..(input_width * (i + 1))].copy_from_slice(input);
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
    op_input: OperationWithIndex,
    op_value_head: OperationWithIndex,
    op_policy_head: OperationWithIndex,
    op_moves_left_head: Option<OperationWithIndex>,
}

struct OperationWithIndex {
    operation: Operation,
    index: c_int,
}

type BatchingModelAndSender<S, A, V, E, Map, Te> = (
    BatchingModel<E, Map, Te>,
    UnboundedSender<StatesToAnalyse<S, A, V>>,
);

pub struct GameAnalyzer<S, A, V, E, Map, Te> {
    batching_model: Arc<BatchingModelAndSender<S, A, V, E, Map, Te>>,
    analysed_state_ordered: CompletedAnalysisOrdered,
    analysed_state_sender: mpsc::UnboundedSender<AnalysisToSend<A, V>>,
}

impl<S, A, V, E, Map, Te> GameAnalyzer<S, A, V, E, Map, Te>
where
    A: Send + 'static,
    V: Send + 'static,
{
    fn new(batching_model: Arc<BatchingModelAndSender<S, A, V, E, Map, Te>>) -> Self {
        let (analysed_state_sender, analyzed_state_receiver) = mpsc::unbounded_channel();

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
    S: Clone + Hash + Unpin,
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
            .unwrap_or_else(|_| debug!("Channel 3 Closed"));

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
    mpsc::UnboundedSender<AnalysisToSend<A, V>>,
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
    S: Hash + Send + 'static,
    A: Clone + Send + 'static,
    V: Value + Send + 'static,
    E: GameEngine<State = S, Action = A, Value = V> + Send + Sync + 'static,
    Map: Mapper<S, A, V, Te> + Send + Sync + 'static,
    Te: Send + 'static,
{
    #[allow(clippy::too_many_arguments)]
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

    #[allow(clippy::too_many_arguments)]
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
                            .unwrap_or_else(|_| debug!("Channel 1 Closed"));
                    }
                    None => {
                        let input =
                            mapper_clone.game_state_to_input(&state_to_analyse, Mode::Infer);
                        states_to_predict_tx
                            .send((id, state_to_analyse, input, unordered_tx, tx))
                            .unwrap_or_else(|_| debug!("Channel 2 Closed"));
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
                        .unwrap_or_else(|_| debug!("Failed to send value in channel."));
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
                        .unwrap_or_else(|_| debug!("Channel 4 Closed"));
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
        analyzed_state_receiver: mpsc::UnboundedReceiver<AnalysisToSend<A, V>>,
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
        receiver: mpsc::UnboundedReceiver<AnalysisToSend<A, V>>,
        capacity: usize,
    ) where
        A: Send + 'static,
        V: Send + 'static,
    {
        tokio::task::spawn(async move {
            let mut analyzed_states_to_tx =
                BinaryHeap::<StateToTransmit<A, V>>::with_capacity(capacity);
            let mut next_id_to_tx: usize = 1;
            let mut receiver = receiver;

            while let Some(analysed_state) = receiver.recv().await {
                let (id, analysis, tx) = analysed_state;
                if id == next_id_to_tx {
                    next_id_to_tx += 1;
                    if tx.send(analysis).is_err() {
                        debug!("Failed to send analysis 1");
                    }

                    while let Some(val) = analyzed_states_to_tx.peek_mut() {
                        if val.id == next_id_to_tx {
                            let state_to_tx = PeekMut::pop(val);
                            next_id_to_tx += 1;
                            if state_to_tx.tx.send(state_to_tx.analysis).is_err() {
                                debug!("Failed to send analysis 2");
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
