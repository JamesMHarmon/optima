use anyhow::Result;
use common::{TranspositionTable, get_env_usize};
use crossbeam::channel::{Receiver as UnboundedReceiver, Sender as UnboundedSender};
use engine::GameEngine;
use engine::Value;
use engine::game_state::GameState;
use half::f16;
use itertools::Itertools;
use log::{debug, info};
use parking_lot::Mutex;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, ParallelBridge, ParallelIterator,
};
use std::collections::HashMap;
use std::future::Future;
use std::os::raw::c_int;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;
use std::sync::Weak;
use std::task::{Context, Poll};
use tensorflow::*;
use tokio::sync::oneshot::{self, Sender};

use super::*;
use ::model::{Analyzer, GameStateAnalysis, Info, ModelInfo, analytics};

#[cfg_attr(feature = "tensorflow_system_alloc", global_allocator)]
#[cfg(feature = "tensorflow_system_alloc")]
static ALLOCATOR: std::alloc::System = std::alloc::System;

#[allow(clippy::type_complexity)]
pub struct TensorflowModel<S, A, P, E, Map, Te> {
    model_info: ModelInfo,
    model_dir: PathBuf,
    batching_model: Mutex<Weak<BatchingModelAndSender<S, A, P, E, Map, Te>>>,
    analysis_request_threads: usize,
    engine: Arc<E>,
    mapper: Arc<Map>,
    batch_size: usize,
    tt_cache_size: usize,
}

impl<S, A, P, E, Map, Te> TensorflowModel<S, A, P, E, Map, Te>
where
    S: Send + Sync + 'static,
    A: Clone + Send + Sync + 'static,
    P: Send + Sync + 'static,
    E: GameEngine<State = S, Action = A, Terminal = P> + Send + Sync + 'static,
    Map: Dimension
        + InputMap<State = S>
        + TranspositionMap<State = S, Action = A, Predictions = P, TranspositionEntry = Te>
        + Send
        + Sync
        + 'static,
    Te: Send + Sync + 'static,
{
    pub fn load(
        model_dir: PathBuf,
        model_info: ModelInfo,
        engine: E,
        mapper: Map,
        tt_cache_size: usize,
    ) -> Result<Self> {
        let batch_size =
            get_env_usize("ANALYSIS_REQUEST_BATCH_SIZE").unwrap_or(ANALYSIS_REQUEST_BATCH_SIZE);
        let analysis_request_threads =
            get_env_usize("ANALYSIS_REQUEST_THREADS").unwrap_or(ANALYSIS_REQUEST_THREADS);

        if std::env::var("TF_CPP_MIN_LOG_LEVEL").is_err() {
            unsafe {
                std::env::set_var("TF_CPP_MIN_LOG_LEVEL", "2");
            }
        }

        info!(
            "Loading model {:?}. Batch Size: {:?}",
            model_info, batch_size
        );

        let mapper = Arc::new(mapper);
        let engine = Arc::new(engine);
        let batching_model = Mutex::new(Weak::new());

        Ok(Self {
            analysis_request_threads,
            batch_size,
            model_info,
            model_dir,
            batching_model,
            engine,
            mapper,
            tt_cache_size,
        })
    }

    fn create_batching_model(
        &self,
        receiver: UnboundedReceiver<StatesToAnalyse<S, A, P>>,
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
            self.model_dir.clone(),
            transposition_table,
            reporter,
            self.batch_size,
            self.analysis_request_threads,
            receiver,
        )
    }
}

impl<S, A, P, E, Map, Te> Analyzer for TensorflowModel<S, A, P, E, Map, Te>
where
    S: GameState + Send + Sync + Unpin + 'static,
    A: Clone + Send + Sync + 'static,
    P: Value + Send + Sync + 'static,
    E: GameEngine<State = S, Action = A, Terminal = P> + Send + Sync + 'static,
    Map: Dimension
        + InputMap<State = S>
        + TranspositionMap<State = S, Action = A, Predictions = P, TranspositionEntry = Te>
        + Clone
        + Send
        + Sync
        + 'static,
    Te: Send + Sync + 'static,
{
    type State = S;
    type Action = A;
    type Predictions = P;
    type Analyzer = GameAnalyzer<S, A, P, E, Map, Te>;

    fn analyzer(&self) -> Self::Analyzer {
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

impl<S, A, P, E, Map, Te> Info for TensorflowModel<S, A, P, E, Map, Te> {
    fn info(&self) -> &ModelInfo {
        &self.model_info
    }
}

pub struct Predictor {
    pub session: Session,
    pub input: OperationWithIndex,
    pub outputs: HashMap<String, OperationWithIndex>,
}

impl Predictor {
    pub fn new(path: &Path) -> Self {
        let mut graph = Graph::new();

        let model: SavedModelBundle =
            SavedModelBundle::load(&SessionOptions::new(), ["serve"], &mut graph, path)
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

        let input = OperationWithIndex::new(
            signature
                .inputs()
                .iter()
                .next()
                .expect("Expected to find input"),
            &graph,
        );
        let outputs: HashMap<String, OperationWithIndex> = signature
            .outputs()
            .iter()
            .map(|signature| {
                (
                    signature.0.to_owned(),
                    OperationWithIndex::new(signature, &graph),
                )
            })
            .collect::<HashMap<_, _>>();
        let session = model.session;

        Self {
            session,
            input,
            outputs,
        }
    }

    fn predict(&self, tensor: &Tensor<f16>) -> Result<AnalysisResults> {
        let mut session_run_args = SessionRunArgs::new();

        session_run_args.add_feed(&self.input.operation, self.input.index, tensor);

        let fetch_tokens = self
            .outputs
            .iter()
            .map(|(name, op)| {
                (
                    name.to_owned(),
                    session_run_args.request_fetch(&op.operation, op.index),
                    op.size,
                )
            })
            .collect_vec();

        self.session
            .run(&mut session_run_args)
            .expect("Expected to be able to run the model session");

        let outputs = fetch_tokens
            .into_iter()
            .map(|(name, fetch_token, size)| {
                (
                    name,
                    AnalysisResult {
                        tensor: session_run_args
                            .fetch(fetch_token)
                            .expect("Expected to be able to load output"),
                        size,
                    },
                )
            })
            .collect::<HashMap<String, AnalysisResult>>();

        Ok(AnalysisResults { outputs })
    }
}

struct AnalysisResult {
    tensor: Tensor<f16>,
    size: usize,
}

struct AnalysisResults {
    outputs: HashMap<String, AnalysisResult>,
}

pub struct OperationWithIndex {
    pub name: String,
    pub operation: Operation,
    pub index: c_int,
    pub size: usize,
}

impl OperationWithIndex {
    fn new(signature: (&String, &TensorInfo), graph: &Graph) -> Self {
        let (name, tensor_info) = signature;
        let shape: Option<Vec<Option<i64>>> = tensor_info.shape().clone().into();
        let size = shape
            .expect("Shape should be defined")
            .into_iter()
            .flatten()
            .product::<i64>() as usize;

        Self {
            name: name.to_owned(),
            operation: graph
                .operation_by_name_required(&tensor_info.name().name)
                .expect("Expected to find input operation"),
            index: tensor_info.name().index,
            size,
        }
    }
}

type BatchingModelAndSender<S, A, P, E, Map, Te> = (
    BatchingModel<E, Map, Te>,
    UnboundedSender<StatesToAnalyse<S, A, P>>,
);

pub struct GameAnalyzer<S, A, P, E, Map, Te> {
    batching_model: Arc<BatchingModelAndSender<S, A, P, E, Map, Te>>,
}

impl<S, A, P, E, Map, Te> GameAnalyzer<S, A, P, E, Map, Te> {
    fn new(batching_model: Arc<BatchingModelAndSender<S, A, P, E, Map, Te>>) -> Self {
        Self { batching_model }
    }
}

impl<S, A, P, E, Map, Te> analytics::GameAnalyzer for GameAnalyzer<S, A, P, E, Map, Te>
where
    S: Clone + Unpin,
    A: Clone,
    E: GameEngine<State = S, Action = A, Terminal = P> + Send + Sync + 'static,
    Te: Send + Sync + 'static,
{
    type State = S;
    type Action = A;
    type Predictions = P;
    type Future = AnalysisFuture<Self::Action, Self::Predictions>;

    fn get_state_analysis(&self, game_state: &S) -> Self::Future {
        let (tx, rx) = oneshot::channel();
        let sender = &self.batching_model.1;
        sender
            .send((game_state.to_owned(), tx))
            .unwrap_or_else(|_| debug!("Channel closed"));

        AnalysisFuture { receiver: rx }
    }
}

/// A future that resolves to a GameStateAnalysis
pub struct AnalysisFuture<A, P> {
    receiver: oneshot::Receiver<GameStateAnalysis<A, P>>,
}

impl<A, P> Future for AnalysisFuture<A, P> {
    type Output = GameStateAnalysis<A, P>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match Pin::new(&mut self.receiver).poll(cx) {
            Poll::Ready(Ok(analysis)) => Poll::Ready(analysis),
            Poll::Ready(Err(_)) => panic!("Analysis channel closed before receiving result"),
            Poll::Pending => Poll::Pending,
        }
    }
}

type StatesToAnalyse<S, A, P> = (S, Sender<GameStateAnalysis<A, P>>);

type StatesToInfer<S, A, P> = (S, Sender<GameStateAnalysis<A, P>>);

struct BatchingModel<E, Map, Te> {
    transposition_table: Arc<Option<TranspositionTable<Te>>>,
    _engine: Arc<E>,
    mapper: Arc<Map>,
    _reporter: Arc<Reporter<Te>>,
}

impl<S, A, P, E, Map, Te> BatchingModel<E, Map, Te>
where
    S: Send + Sync + 'static,
    A: Clone + Send + 'static,
    P: Send + 'static,
    E: GameEngine<State = S, Action = A, Terminal = P> + Send + Sync + 'static,
    Map: Dimension
        + InputMap<State = S>
        + TranspositionMap<State = S, Action = A, Predictions = P, TranspositionEntry = Te>
        + Send
        + Sync
        + 'static,
    Te: Send + 'static,
{
    #[allow(clippy::too_many_arguments)]
    fn new(
        engine: Arc<E>,
        mapper: Arc<Map>,
        model_dir: PathBuf,
        transposition_table: Arc<Option<TranspositionTable<Te>>>,
        reporter: Arc<Reporter<Te>>,
        batch_size: usize,
        analysis_request_threads: usize,
        states_to_analyse_receiver: UnboundedReceiver<StatesToAnalyse<S, A, P>>,
    ) -> Self {
        let inner_self = Self {
            transposition_table,
            _reporter: reporter.clone(),
            _engine: engine.clone(),
            mapper,
        };

        inner_self.create_analysis_tasks(
            engine,
            reporter,
            model_dir,
            states_to_analyse_receiver,
            batch_size,
            analysis_request_threads,
        );

        inner_self
    }

    fn listen_then_transpose_or_infer(
        engine: Arc<E>,
        mapper: Arc<Map>,
        transposition_table: Arc<Option<TranspositionTable<Te>>>,
        reporter: Arc<Reporter<Te>>,
        receiver: UnboundedReceiver<StatesToAnalyse<S, A, P>>,
        states_to_predict_tx: UnboundedSender<StatesToInfer<S, A, P>>,
    ) {
        let engine = &*engine;
        let mapper = &*mapper;
        let transposition_table = &*transposition_table;
        let reporter = &*reporter;
        let states_to_predict_tx = &states_to_predict_tx;

        rayon::scope_fifo(move |s| {
            while let Ok((state_to_analyse, tx)) = receiver.recv() {
                s.spawn_fifo(move |_| {
                    if let Some(analysis) = Self::try_immediate_analysis(
                        &state_to_analyse,
                        transposition_table,
                        engine,
                        mapper,
                        reporter,
                    ) {
                        tx.send(analysis)
                            .unwrap_or_else(|_| debug!("Channel closed"));
                    } else {
                        states_to_predict_tx
                            .send((state_to_analyse, tx))
                            .unwrap_or_else(|_| debug!("Channel closed"));
                    }
                });
            }
        });

        debug!("Exiting listen_then_transpose_or_infer");
    }

    #[allow(clippy::too_many_arguments)]
    fn create_analysis_tasks(
        &self,
        engine: Arc<E>,
        reporter: Arc<Reporter<Te>>,
        model_dir: PathBuf,
        receiver: UnboundedReceiver<StatesToAnalyse<S, A, P>>,
        batch_size: usize,
        analysis_request_threads: usize,
    ) {
        let (states_to_predict_tx, states_to_predict_rx) = crossbeam::channel::unbounded();

        let transposition_table_clone = self.transposition_table.clone();
        let mapper_clone = self.mapper.clone();
        let reporter_clone = reporter.clone();

        std::thread::spawn(move || {
            Self::listen_then_transpose_or_infer(
                engine,
                mapper_clone.clone(),
                transposition_table_clone.clone(),
                reporter_clone.clone(),
                receiver,
                states_to_predict_tx,
            );
        });

        let predictor = Arc::new(Predictor::new(&model_dir));
        let filling_states_to_analyze = Arc::new(std::sync::Mutex::new(()));

        for _ in 0..analysis_request_threads {
            let mapper = self.mapper.clone();
            let reporter = reporter.clone();
            let predictor = predictor.clone();
            let states_to_predict_rx = states_to_predict_rx.clone();
            let filling_states_to_analyze = filling_states_to_analyze.clone();
            let transposition_table = self.transposition_table.clone();

            std::thread::spawn(move || {
                let dimensions = mapper.dimensions();
                let input_len = dimensions.iter().product::<u64>() as usize;
                let mut tensor_pool = TensorPool::<f16>::new(dimensions);

                loop {
                    // Lock when getting states to analyze so that the pulled states maintain order. Otherwise when reordering later, any missed states will need to be waited for.
                    let lock = filling_states_to_analyze.lock().unwrap();
                    let states_to_analyse = states_to_predict_rx.recv_up_to(batch_size);
                    drop(lock);

                    // If states is empty then the channel tx has been closed.
                    if states_to_analyse.is_empty() {
                        return;
                    }

                    let tensor: &mut Tensor<f16> =
                        tensor_pool.get(states_to_analyse.len(), half::f16::ZERO);

                    reporter.set_batch_size(states_to_analyse.len());

                    states_to_analyse
                        .iter()
                        .zip(tensor.chunks_mut(input_len))
                        .par_bridge()
                        .for_each(|((state_to_analyse, _), tensor_chunk)| {
                            mapper.game_state_to_input(state_to_analyse, tensor_chunk, Mode::Infer);
                        });

                    let predictions = predictor
                        .predict(tensor)
                        .expect("Expected predict to be successful");

                    let mapper = mapper.clone();
                    let transposition_table = transposition_table.clone();

                    Self::process_predictions(
                        predictions,
                        states_to_analyse,
                        transposition_table,
                        mapper,
                    );
                }
            });
        }
    }

    fn process_predictions(
        predictions: AnalysisResults,
        states_to_analyse: Vec<StatesToInfer<S, A, P>>,
        transposition_table: Arc<Option<TranspositionTable<Te>>>,
        mapper: Arc<Map>,
    ) {
        states_to_analyse
            .into_par_iter()
            .enumerate()
            .for_each(|(i, (game_state, tx))| {
                let outputs = predictions
                    .outputs
                    .iter()
                    .map(|(k, r)| (k.to_owned(), &r.tensor[(i * r.size)..((i + 1) * r.size)]))
                    .collect::<HashMap<String, &[f16]>>();

                let mapper = &*mapper;
                let transposition_table = &*transposition_table;
                let transposition_table_entry =
                    mapper.map_output_to_transposition_entry(&game_state, outputs);

                let analysis = mapper
                    .map_transposition_entry_to_analysis(&game_state, &transposition_table_entry);

                Self::set_transposition_entry(
                    &game_state,
                    transposition_table,
                    transposition_table_entry,
                    mapper,
                );

                tx.send(analysis)
                    .unwrap_or_else(|_| debug!("Channel closed"));
            });
    }

    fn set_transposition_entry(
        game_state: &S,
        transposition_table: &Option<TranspositionTable<Te>>,
        transposition_table_entry: Te,
        mapper: &Map,
    ) {
        if let Some(transposition_table) = transposition_table {
            let transposition_key = mapper.get_transposition_key(game_state);
            transposition_table.set(transposition_key, transposition_table_entry);
        }
    }

    fn try_immediate_analysis(
        game_state: &S,
        transposition_table: &Option<TranspositionTable<Te>>,
        engine: &E,
        mapper: &Map,
        reporter: &Reporter<Te>,
    ) -> Option<GameStateAnalysis<A, P>> {
        if let Some(value) = engine.terminal_state(game_state) {
            reporter.set_terminal();

            return Some(GameStateAnalysis::new(Vec::new(), value));
        }

        if let Some(transposition_table) = transposition_table {
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

struct TensorPool<T: TensorType> {
    tensors: Vec<Tensor<T>>,
    dimensions: [u64; 3],
}

const BATCH_SIZE_INCR: usize = 32;

impl<T: TensorType> TensorPool<T> {
    fn new(dimensions: [u64; 3]) -> Self {
        Self {
            tensors: vec![],
            dimensions,
        }
    }

    fn get(&mut self, size: usize, fill: T) -> &mut Tensor<T> {
        let idx = (size - 1) / BATCH_SIZE_INCR;
        let tensors = &mut self.tensors;
        while tensors.len() <= idx {
            tensors.push(Tensor::new(&[
                ((tensors.len() + 1) * BATCH_SIZE_INCR) as u64,
                self.dimensions[0],
                self.dimensions[1],
                self.dimensions[2],
            ]));
        }

        let tensor = &mut tensors[idx];

        tensor[..].fill(fill);

        tensor
    }
}
