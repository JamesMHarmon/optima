use std::pin::Pin;
use std::sync::Arc;
use chashmap::CHashMap;
use crossbeam_queue::ArrayQueue;
use std::hash::{Hash,Hasher};
use super::analytics::GameStateAnalysis;
use super::analytics::GameAnalyzer;
use super::model::TrainOptions;
use super::position_metrics::PositionMetrics;
use super::model::Model;
use super::model_info::ModelInfo;
use futures::future::FutureExt;
use std::future::Future;

use failure::Error;

const CACHE_SIZE: usize = 5_000_000;
const CACHE_BUFFER: usize = 5_000;

pub struct AnalysisCacheQueueModel<M>
where
    M: Model,
    <<M as Model>::Analyzer as GameAnalyzer>::State: Hash,
    <<M as Model>::Analyzer as GameAnalyzer>::Action: Clone,
    <<M as Model>::Analyzer as GameAnalyzer>::Value: Clone
{
    model: M,
    cache: Arc<CHashMap<u64,GameStateAnalysis<<<M as Model>::Analyzer as GameAnalyzer>::Action,<<M as Model>::Analyzer as GameAnalyzer>::Value>>>,
    queue: Arc<ArrayQueue<u64>>
}

pub fn cache<M>(model: M) -> AnalysisCacheQueueModel<M>
where
    M: Model,
    <<M as Model>::Analyzer as GameAnalyzer>::State: Hash,
    <<M as Model>::Analyzer as GameAnalyzer>::Action: Clone,
    <<M as Model>::Analyzer as GameAnalyzer>::Value: Clone
{
    AnalysisCacheQueueModel {
        model,
        cache: Arc::new(CHashMap::with_capacity(CACHE_SIZE)),
        queue: Arc::new(ArrayQueue::new(CACHE_SIZE))
    }
}

impl<M> Model for AnalysisCacheQueueModel<M>
where
    M: Model,
    M::State: Send + Sync + 'static,
    M::Action: Clone + Send + Sync + 'static,
    M::Value: Clone + Send + Sync + 'static,
    <<M as Model>::Analyzer as GameAnalyzer>::State: PartialEq + Hash + Send + Sync,
    <<M as Model>::Analyzer as GameAnalyzer>::Action: Clone + Send + Sync,
    <<M as Model>::Analyzer as GameAnalyzer>::Value: Clone + Send + Sync + 'static,
    <<M as Model>::Analyzer as GameAnalyzer>::Future: Send + 'static
{
    type State = M::State;
    type Action = M::Action;
    type Value = M::Value;
    type Analyzer = AnalysisCacheAnalyzer<M::Analyzer>;

    fn get_model_info(&self) -> &ModelInfo {
        self.model.get_model_info()
    }

    fn train<I: Iterator<Item=PositionMetrics<Self::State,Self::Action,Self::Value>>>(&self, target_model_info: &ModelInfo, sample_metrics: I, options: &TrainOptions) -> Result<(), Error> {
        M::train(&self.model, target_model_info, sample_metrics, options)
    }

    fn get_game_state_analyzer(&self) -> Self::Analyzer {
        let analyzer = M::get_game_state_analyzer(&self.model);

        AnalysisCacheAnalyzer {
            analyzer,
            cache: self.cache.clone(),
            queue: self.queue.clone(),
        }
    }
}

pub struct AnalysisCacheAnalyzer<Analyzer>
where
    Analyzer: GameAnalyzer,
    Analyzer::State: Hash,
    Analyzer::Action: Clone,
    Analyzer::Value: Clone
{
    analyzer: Analyzer,
    cache: Arc<CHashMap<u64,GameStateAnalysis<Analyzer::Action,Analyzer::Value>>>,
    queue: Arc<ArrayQueue<u64>>
}

impl<Analyzer> GameAnalyzer for AnalysisCacheAnalyzer<Analyzer>
where
    Analyzer: GameAnalyzer,
    Analyzer::Future: Send + 'static,
    Analyzer::State: Hash + Send + Sync + 'static,
    Analyzer::Action: Clone + Send + Sync + 'static,
    Analyzer::Value: Clone + Send + Sync + 'static
{
    type State = Analyzer::State;
    type Action = Analyzer::Action;
    type Value = Analyzer::Value;
    type Future = Pin<Box<dyn Future<Output=GameStateAnalysis<Self::Action,Analyzer::Value>> + Send>>;

    fn get_state_analysis(&self, game_state: &Self::State) -> Self::Future {
        let cache = &self.cache;
        let game_state_hash = calculate_hash(game_state);

        if let Some(analysis) = cache.get(&game_state_hash) {
            let analysis = analysis.clone();
            return futures::future::ready(analysis).boxed();
        }

        let cache = cache.clone();
        let queue = self.queue.clone();
        Analyzer::get_state_analysis(&self.analyzer, &game_state).map(move |analysis| {
            if cache.len() >= CACHE_SIZE - CACHE_BUFFER {
                if let Ok(item_to_remove) = queue.pop() {
                    cache.remove(&item_to_remove);
                }
            }

            cache.insert(game_state_hash, analysis.clone());
            queue.push(game_state_hash).unwrap();

            analysis
        }).boxed()
    }
}

fn calculate_hash<T: Hash>(t: &T) -> u64 {
    let mut s = IdentityHasher::new();
    t.hash(&mut s);
    s.finish()
}

struct IdentityHasher {
    hash: u64
}

impl Hasher for IdentityHasher {
    fn finish(&self) -> u64 {
        self.hash
    }

    fn write(&mut self, _bytes: &[u8]) {
        panic!("Not Supported: write for identity hash")
    }

    fn write_u64(&mut self, i: u64) {
        self.hash = i;
    }
}

impl IdentityHasher {
    fn new() -> Self {
        Self {
            hash: 0
        }
    }
}
