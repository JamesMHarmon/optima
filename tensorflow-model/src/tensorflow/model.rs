use anyhow::Result;
use common::{TranspositionTable, get_env_usize};
use dashmap::DashMap;
use dashmap::mapref::entry::Entry;
use half::f16;
use log::info;
use parking_lot::Mutex;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, ParallelBridge, ParallelIterator,
};
use smallvec::SmallVec;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Weak};
use tensorflow::Tensor;

use super::predictor::{AnalysisResults, Predictor};
use super::tensor_pool::TensorPool;
use super::*;
use ::model::{Analyzer, GameStateAnalysis, Info, ModelInfo, analytics};

#[cfg_attr(feature = "tensorflow_system_alloc", global_allocator)]
#[cfg(feature = "tensorflow_system_alloc")]
static ALLOCATOR: std::alloc::System = std::alloc::System;

pub struct TensorflowModel<S, A, P, Map, Te> {
    model_info: ModelInfo,
    model_dir: PathBuf,
    batching_model: BatchingModelRef<S, A, P, Map, Te>,
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

type ResultsTx<A, P> = crossbeam::channel::Sender<(usize, GameStateAnalysis<A, P>)>;
type ResultsRx<A, P> = crossbeam::channel::Receiver<(usize, GameStateAnalysis<A, P>)>;
type WaiterList<A, P> = SmallVec<[(usize, ResultsTx<A, P>); 2]>;
type BatchingModelRef<S, A, P, Map, Te> = Mutex<Weak<BatchingModel<S, A, P, Map, Te>>>;

/// Tracks in-flight analysis requests to coalesce duplicate requests.
/// Maps transposition key to list of waiters for that state's analysis result.
struct InFlightRequests<A, P> {
    requests: DashMap<u64, InFlightEntry<A, P>>,
}

struct InFlightEntry<A, P> {
    waiters: WaiterList<A, P>,
}

impl<A, P> InFlightRequests<A, P> {
    fn new() -> Self {
        Self {
            requests: DashMap::new(),
        }
    }

    /// Register a request.
    /// Returns true if this key was already in-flight (duplicate).
    fn try_register(&self, key: u64, request_id: usize, tx: ResultsTx<A, P>) -> bool {
        match self.requests.entry(key) {
            Entry::Occupied(mut entry) => {
                entry.get_mut().waiters.push((request_id, tx));
                true
            }
            Entry::Vacant(entry) => {
                let mut waiters = SmallVec::new();
                waiters.push((request_id, tx));
                entry.insert(InFlightEntry { waiters });
                false
            }
        }
    }

    /// Remove and return all waiters for this state's analysis
    fn take_waiters(&self, key: u64) -> Option<WaiterList<A, P>> {
        self.requests.remove(&key).map(|(_, v)| v.waiters)
    }
}

pub struct GameAnalyzer<S, A, P, Map, Te> {
    batching_model: Arc<BatchingModel<S, A, P, Map, Te>>,
    results_tx: ResultsTx<A, P>,
    results_rx: ResultsRx<A, P>,
}

impl<S, A, P, Map, Te> GameAnalyzer<S, A, P, Map, Te> {
    fn new(batching_model: Arc<BatchingModel<S, A, P, Map, Te>>) -> Self {
        let (results_tx, results_rx) = crossbeam::channel::unbounded();
        Self {
            batching_model,
            results_tx,
            results_rx,
        }
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
}

impl<S, A, P, Map, Te> analytics::GameAnalyzer for GameAnalyzer<S, A, P, Map, Te>
where
    Map: TranspositionMap<State = S, Action = A, Predictions = P, TranspositionEntry = Te>,
    Te: Send + 'static,
    S: Clone,
    A: Clone,
    P: Clone,
{
    type State = S;
    type Action = A;
    type Predictions = P;
    type RequestId = usize;

    fn analyze(&self, request_id: usize, game_state: &S) {
        if let Some(analysis) = self.try_immediate_analysis(game_state) {
            let _ = self.results_tx.send((request_id, analysis));
            return;
        }

        self.batching_model
            .enqueue(game_state.to_owned(), request_id, self.results_tx.clone());
    }

    fn recv(&self) -> (usize, GameStateAnalysis<A, P>) {
        self.results_rx
            .recv()
            .expect("Results channel closed before receiving result")
    }
}

type StatesToInfer<S> = (S, u64); // State and its transposition key

type StatesToAnalyzeTx<S> = crossbeam::channel::Sender<StatesToInfer<S>>;
type StatesToAnalyzeRx<S> = crossbeam::channel::Receiver<StatesToInfer<S>>;

struct BatchingModel<S, A, P, Map, Te> {
    transposition_table: Arc<Option<TranspositionTable<Te>>>,
    in_flight_requests: Arc<InFlightRequests<A, P>>,
    mapper: Arc<Map>,
    reporter: Arc<Reporter<Te>>,
    states_to_analyze_tx: StatesToAnalyzeTx<S>,
}

impl<S, A, P, Map, Te> BatchingModel<S, A, P, Map, Te>
where
    Map: TranspositionMap<State = S, Action = A, Predictions = P, TranspositionEntry = Te>,
    Te: Send + 'static,
{
    fn enqueue(&self, state_to_analyse: S, request_id: usize, tx: ResultsTx<A, P>) {
        let key = self.mapper.get_transposition_key(&state_to_analyse);
        let is_duplicate = self.in_flight_requests.try_register(key, request_id, tx);

        if is_duplicate {
            self.reporter.set_predict_in_flight();
        } else {
            self.reporter.set_predict_needs_infer();
            let _ = self.states_to_analyze_tx.send((state_to_analyse, key));
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

        let queue_capacity = batch_size * analysis_request_threads;
        let (states_to_analyze_tx, states_to_analyze_rx) =
            crossbeam::channel::bounded(queue_capacity);

        let inner_self = Self {
            transposition_table,
            in_flight_requests: in_flight_requests.clone(),
            reporter: reporter.clone(),
            mapper,
            states_to_analyze_tx,
        };

        inner_self.create_analysis_tasks(
            reporter,
            model_dir,
            batch_size,
            analysis_request_threads,
            in_flight_requests,
            states_to_analyze_rx,
        );

        inner_self
    }

    fn create_analysis_tasks(
        &self,
        reporter: Arc<Reporter<Te>>,
        model_dir: PathBuf,
        batch_size: usize,
        analysis_request_threads: usize,
        in_flight_requests: Arc<InFlightRequests<A, P>>,
        states_to_analyze_rx: StatesToAnalyzeRx<S>,
    ) {
        let predictor = Arc::new(Predictor::new(&model_dir));

        for _ in 0..analysis_request_threads {
            let mapper = self.mapper.clone();
            let reporter = reporter.clone();
            let predictor = predictor.clone();
            let transposition_table = self.transposition_table.clone();
            let in_flight_requests = in_flight_requests.clone();
            let states_to_analyze_rx = states_to_analyze_rx.clone();

            std::thread::spawn(move || {
                let dimensions = mapper.dimensions();
                let input_len = dimensions.iter().product::<u64>() as usize;
                let mut tensor_pool = TensorPool::<f16>::new(dimensions);

                loop {
                    let states_to_analyse = states_to_analyze_rx.recv_up_to(batch_size);

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

                if let Some(waiters) = in_flight_requests.take_waiters(key) {
                    for (request_id, tx) in waiters {
                        let _ = tx.send((request_id, analysis.clone()));
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
}
