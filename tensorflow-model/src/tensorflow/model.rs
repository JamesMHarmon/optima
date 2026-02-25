use anyhow::Result;
use common::{TranspositionTable, get_env_usize};
use dashmap::DashMap;
use dashmap::mapref::entry::Entry;
use half::f16;
use itertools::Itertools;
use log::info;
use parking_lot::Mutex;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, ParallelBridge, ParallelIterator,
};
use smallvec::SmallVec;
use std::collections::HashMap;
use std::os::raw::c_int;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Weak};
use tensorflow::*;

use super::*;
use ::model::{Analyzer, GameStateAnalysis, Info, ModelInfo, analytics};

#[cfg_attr(feature = "tensorflow_system_alloc", global_allocator)]
#[cfg(feature = "tensorflow_system_alloc")]
static ALLOCATOR: std::alloc::System = std::alloc::System;

#[allow(clippy::type_complexity)]
pub struct TensorflowModel<S, A, P, Map, Te> {
    model_info: ModelInfo,
    model_dir: PathBuf,
    batching_model: Mutex<Weak<BatchingModel<S, A, P, Map, Te>>>,
    analysis_request_threads: usize,
    mapper: Arc<Map>,
    batch_size: usize,
    tt_cache_size: usize,
}

impl<S, A, P, Map, Te> TensorflowModel<S, A, P, Map, Te> {
    pub fn load(
        model_dir: PathBuf,
        model_info: ModelInfo,
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
        let batching_model = Mutex::new(Weak::new());

        Ok(Self {
            analysis_request_threads,
            batch_size,
            model_info,
            model_dir,
            batching_model,
            mapper,
            tt_cache_size,
        })
    }

    fn create_batching_model(&self) -> BatchingModel<S, A, P, Map, Te>
    where
        S: Send + Sync + 'static,
        A: Clone + Send + 'static,
        P: Clone + Send + 'static,
        Map: Dimension
            + InputMap<State = S>
            + TranspositionMap<State = S, Action = A, Predictions = P, TranspositionEntry = Te>
            + Send
            + Sync
            + 'static,
        Te: Send + 'static,
    {
        let transposition_table = Arc::new(if self.tt_cache_size > 0 {
            Some(TranspositionTable::new(self.tt_cache_size))
        } else {
            None
        });

        let reporter = Arc::new(Reporter::new(transposition_table.clone()));

        BatchingModel::new(
            self.mapper.clone(),
            self.model_dir.clone(),
            transposition_table,
            reporter,
            self.batch_size,
            self.analysis_request_threads,
        )
    }
}

impl<S, A, P, Map, Te> Analyzer for TensorflowModel<S, A, P, Map, Te>
where
    S: Clone + Send + Sync + 'static,
    A: Clone + Send + 'static,
    P: Clone + Send + 'static,
    Map: Dimension
        + InputMap<State = S>
        + TranspositionMap<State = S, Action = A, Predictions = P, TranspositionEntry = Te>
        + Send
        + Sync
        + 'static,
    Te: Send + 'static,
{
    type State = S;
    type Action = A;
    type Predictions = P;
    type Analyzer = GameAnalyzer<S, A, P, Map, Te>;

    fn analyzer(&self) -> Self::Analyzer {
        let batching_model_ref = &mut *self.batching_model.lock();
        let batching_model = batching_model_ref.upgrade();

        let batching_model = match batching_model {
            None => {
                let batching_model_arc = Arc::new(self.create_batching_model());

                *batching_model_ref = Arc::downgrade(&batching_model_arc);

                batching_model_arc
            }
            Some(model_sender) => model_sender,
        };

        Self::Analyzer::new(batching_model)
    }
}

impl<S, A, P, Map, Te> Info for TensorflowModel<S, A, P, Map, Te> {
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

type PredictResponseTx<A, P> = crossbeam::channel::Sender<GameStateAnalysis<A, P>>;

/// Tracks in-flight analysis requests to coalesce duplicate requests.
/// Maps transposition key to list of channels waiting for that state's analysis.
/// For prefetch, we insert an entry but do not store any channel.
struct InFlightRequests<A, P> {
    requests: DashMap<u64, InFlightEntry<A, P>>,
}

struct InFlightEntry<A, P> {
    waiters: SmallVec<[PredictResponseTx<A, P>; 2]>,
    high_enqueued: bool,
}

impl<A, P> InFlightRequests<A, P> {
    fn new() -> Self {
        Self {
            requests: DashMap::new(),
        }
    }

    /// Register a Predict request.
    /// Returns (is_duplicate, should_enqueue_high).
    fn try_register_predict(&self, key: u64, channel: PredictResponseTx<A, P>) -> (bool, bool) {
        match self.requests.entry(key) {
            Entry::Occupied(mut entry) => {
                let in_flight = entry.get_mut();
                in_flight.waiters.push(channel);

                if in_flight.high_enqueued {
                    (true, false)
                } else {
                    in_flight.high_enqueued = true;
                    (true, true)
                }
            }
            Entry::Vacant(entry) => {
                let mut waiters = SmallVec::new();
                waiters.push(channel);
                entry.insert(InFlightEntry {
                    waiters,
                    high_enqueued: true,
                });
                (false, true)
            }
        }
    }

    /// Try to register a Prefetch request.
    /// Returns true if this key was already in-flight.
    fn try_register_prefetch(&self, key: u64) -> bool {
        match self.requests.entry(key) {
            Entry::Occupied(_entry) => true,
            Entry::Vacant(entry) => {
                entry.insert(InFlightEntry {
                    waiters: SmallVec::new(),
                    high_enqueued: false,
                });
                false
            }
        }
    }

    /// Remove and return all channels waiting for this state's analysis
    fn take_channels(&self, key: u64) -> Option<SmallVec<[PredictResponseTx<A, P>; 2]>> {
        self.requests.remove(&key).map(|(_, v)| v.waiters)
    }
}

pub struct GameAnalyzer<S, A, P, Map, Te> {
    batching_model: Arc<BatchingModel<S, A, P, Map, Te>>,
}

impl<S, A, P, Map, Te> GameAnalyzer<S, A, P, Map, Te> {
    fn new(batching_model: Arc<BatchingModel<S, A, P, Map, Te>>) -> Self {
        Self { batching_model }
    }
}

impl<S, A, P, Map, Te> GameAnalyzer<S, A, P, Map, Te>
where
    Map: TranspositionMap<State = S, Action = A, Predictions = P, TranspositionEntry = Te>,
    Te: Send + 'static,
{
    fn try_immediate_analysis(&self, game_state: &S) -> Option<GameStateAnalysis<A, P>> {
        let transposition_table = &*self.batching_model.transposition_table;
        let mapper = &*self.batching_model.mapper;
        let reporter = &*self.batching_model.reporter;

        if let Some(transposition_entry) = transposition_table
            .as_ref()
            .and_then(|tt| tt.get(mapper.get_transposition_key(game_state)))
        {
            let analysis =
                mapper.map_transposition_entry_to_analysis(game_state, &*transposition_entry);
            drop(transposition_entry);
            reporter.set_cache_hit();

            return Some(analysis);
        }

        if self.batching_model.transposition_table.is_some() {
            reporter.set_cache_miss();
        }

        None
    }

    fn has_analysis(&self, game_state: &S) -> bool {
        let transposition_table = &*self.batching_model.transposition_table;
        let mapper = &*self.batching_model.mapper;

        transposition_table
            .as_ref()
            .and_then(|tt| tt.get(mapper.get_transposition_key(game_state)))
            .is_some()
    }
}

// @TODO: Analyze and prefetch should probably check the TT before sending to the batching model, to avoid unnecessary latency when the TT can immediately answer the request.
impl<S, A, P, Map, Te> analytics::GameAnalyzer for GameAnalyzer<S, A, P, Map, Te>
where
    Map: TranspositionMap<State = S, Action = A, Predictions = P, TranspositionEntry = Te>,
    Te: Send + 'static,
    S: Clone,
{
    type State = S;
    type Action = A;
    type Predictions = P;

    fn analyze(&self, game_state: &S) -> GameStateAnalysis<A, P> {
        if let Some(entry) = self.try_immediate_analysis(game_state) {
            return entry;
        }

        let (tx, rx) = crossbeam::channel::bounded(1);
        self.batching_model
            .enqueue_predict(game_state.to_owned(), tx);

        rx.recv()
            .expect("Analysis channel closed before receiving result")
    }

    fn prefetch(&self, game_state: &Self::State) {
        if self.has_analysis(game_state) {
            return;
        }

        self.batching_model.enqueue_prefetch(game_state.to_owned());
    }
}

type StatesToInfer<S> = (S, u64); // State and its transposition key

type StatesToPredictTx<S> = crossbeam::channel::Sender<StatesToInfer<S>>;
type StatesToPredictRx<S> = crossbeam::channel::Receiver<StatesToInfer<S>>;
type StatesToPrefetchTx<S> = crossbeam::channel::Sender<StatesToInfer<S>>;
type StatesToPrefetchRx<S> = crossbeam::channel::Receiver<StatesToInfer<S>>;

struct BatchingModel<S, A, P, Map, Te> {
    transposition_table: Arc<Option<TranspositionTable<Te>>>,
    in_flight_requests: Arc<InFlightRequests<A, P>>,
    mapper: Arc<Map>,
    reporter: Arc<Reporter<Te>>,
    states_to_predict_tx: StatesToPredictTx<S>,
    states_to_prefetch_tx: StatesToPrefetchTx<S>,
}

impl<S, A, P, Map, Te> BatchingModel<S, A, P, Map, Te>
where
    Map: TranspositionMap<State = S, Action = A, Predictions = P, TranspositionEntry = Te>,
    Te: Send + 'static,
{
    fn enqueue_predict(&self, state_to_analyse: S, channel: PredictResponseTx<A, P>) {
        let key = self.mapper.get_transposition_key(&state_to_analyse);
        let (is_duplicate, should_enqueue_high) =
            self.in_flight_requests.try_register_predict(key, channel);

        if is_duplicate {
            self.reporter.set_predict_in_flight();
        } else {
            self.reporter.set_predict_needs_infer();
        }

        // Enqueue exactly one high-priority inference per key.
        // If this key was first queued via Prefetch, the first Predict
        // promotes it by enqueuing to the high-priority queue once.
        if should_enqueue_high {
            let _ = self.states_to_predict_tx.send((state_to_analyse, key));
        }
    }

    fn enqueue_prefetch(&self, state_to_analyse: S) {
        let key = self.mapper.get_transposition_key(&state_to_analyse);
        let is_duplicate = self.in_flight_requests.try_register_prefetch(key);
        if !is_duplicate {
            let _ = self.states_to_prefetch_tx.send((state_to_analyse, key));
        }
    }
}

impl<S, A, P, Map, Te> BatchingModel<S, A, P, Map, Te>
where
    S: Send + Sync + 'static,
    A: Clone + Send + 'static,
    P: Clone + Send + 'static,
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
        mapper: Arc<Map>,
        model_dir: PathBuf,
        transposition_table: Arc<Option<TranspositionTable<Te>>>,
        reporter: Arc<Reporter<Te>>,
        batch_size: usize,
        analysis_request_threads: usize,
    ) -> Self {
        let in_flight_requests = Arc::new(InFlightRequests::new());

        let (states_to_predict_tx, states_to_predict_rx) = crossbeam::channel::unbounded();
        let (states_to_prefetch_tx, states_to_prefetch_rx) = crossbeam::channel::unbounded();

        let inner_self = Self {
            transposition_table,
            in_flight_requests: in_flight_requests.clone(),
            reporter: reporter.clone(),
            mapper,
            states_to_predict_tx,
            states_to_prefetch_tx,
        };

        inner_self.create_analysis_tasks(
            reporter,
            model_dir,
            batch_size,
            analysis_request_threads,
            in_flight_requests,
            states_to_predict_rx,
            states_to_prefetch_rx,
        );

        inner_self
    }

    #[allow(clippy::too_many_arguments)]
    fn create_analysis_tasks(
        &self,
        reporter: Arc<Reporter<Te>>,
        model_dir: PathBuf,
        batch_size: usize,
        analysis_request_threads: usize,
        in_flight_requests: Arc<InFlightRequests<A, P>>,
        states_to_predict_rx: StatesToPredictRx<S>,
        states_to_prefetch_rx: StatesToPrefetchRx<S>,
    ) {
        let predictor = Arc::new(Predictor::new(&model_dir));

        for _ in 0..analysis_request_threads {
            let mapper = self.mapper.clone();
            let reporter = reporter.clone();
            let predictor = predictor.clone();
            let transposition_table = self.transposition_table.clone();
            let in_flight_requests = in_flight_requests.clone();
            let states_to_predict_rx = states_to_predict_rx.clone();
            let states_to_prefetch_rx = states_to_prefetch_rx.clone();

            std::thread::spawn(move || {
                let dimensions = mapper.dimensions();
                let input_len = dimensions.iter().product::<u64>() as usize;
                let mut tensor_pool = TensorPool::<f16>::new(dimensions);

                loop {
                    let states_to_analyse = Self::recv_up_to_prioritized(
                        &states_to_predict_rx,
                        &states_to_prefetch_rx,
                        batch_size,
                    );

                    // If states is empty then the high-priority channel has been closed.
                    if states_to_analyse.is_empty() {
                        return;
                    }

                    let tensor: &mut Tensor<f16> =
                        tensor_pool.get(states_to_analyse.len(), half::f16::ZERO);

                    reporter.set_batch_size(states_to_analyse.len());
                    reporter.set_analyzed_nodes(states_to_analyse.len());

                    states_to_analyse
                        .iter()
                        .zip(tensor.chunks_mut(input_len))
                        .par_bridge()
                        .for_each(|((state_to_analyse, _key), tensor_chunk)| {
                            mapper.game_state_to_input(state_to_analyse, tensor_chunk, Mode::Infer);
                        });

                    let predictions = predictor
                        .predict(tensor)
                        .expect("Expected predict to be successful");

                    let mapper = mapper.clone();
                    let transposition_table = transposition_table.clone();
                    let in_flight_requests = in_flight_requests.clone();

                    Self::process_predictions(
                        predictions,
                        states_to_analyse,
                        transposition_table,
                        mapper,
                        in_flight_requests,
                    );
                }
            });
        }
    }

    fn process_predictions(
        predictions: AnalysisResults,
        states_to_analyse: Vec<StatesToInfer<S>>,
        transposition_table: Arc<Option<TranspositionTable<Te>>>,
        mapper: Arc<Map>,
        in_flight_requests: Arc<InFlightRequests<A, P>>,
    ) {
        states_to_analyse
            .into_par_iter()
            .enumerate()
            .for_each(|(i, (game_state, key))| {
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

                // Send result to all channels waiting for this state
                if let Some(channels) = in_flight_requests.take_channels(key) {
                    for channel in channels {
                        let _ = channel.send(analysis.clone());
                    }
                }
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

    fn recv_up_to_prioritized(
        high: &StatesToPredictRx<S>,
        low: &StatesToPrefetchRx<S>,
        limit: usize,
    ) -> Vec<StatesToInfer<S>> {
        let mut out = Vec::with_capacity(limit);

        // Block for the first item and require it to come from the high-priority queue.
        // This intentionally deprioritizes prefetch so it never starts a batch by itself.
        let Ok(v) = high.recv() else {
            return out;
        };
        out.push(v);

        // Drain high-priority first.
        let remaining = limit.saturating_sub(out.len());
        out.extend(high.try_iter().take(remaining));

        // Fill remaining capacity from low-priority.
        let remaining = limit.saturating_sub(out.len());
        out.extend(low.try_iter().take(remaining));

        out
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
