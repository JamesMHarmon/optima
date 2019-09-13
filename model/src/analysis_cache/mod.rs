use std::sync::Arc;
use chashmap::CHashMap;
use std::hash::Hash;
use super::analytics::GameStateAnalysis;
use super::analytics::GameAnalyzer;
use super::model::TrainOptions;
use super::position_metrics::PositionMetrics;
use super::model::Model;
use super::model_info::ModelInfo;
use futures::future::FutureExt;
use std::marker::PhantomData;
use std::future::Future;

use failure::Error;

pub trait ShouldCache {
    type State;

    fn should_cache(game_state: &Self::State) -> bool;
}

pub struct AnalysisCacheModel<C,M>
where
    M: Model,
    C: ShouldCache<State=M::State> + Send,
    <<M as Model>::Analyzer as GameAnalyzer>::State: PartialEq + Hash,
    <<M as Model>::Analyzer as GameAnalyzer>::Action: Clone,
{
    model: M,
    phantom: PhantomData<C>,
    cache: Arc<CHashMap<<<M as Model>::Analyzer as GameAnalyzer>::State,GameStateAnalysis<<<M as Model>::Analyzer as GameAnalyzer>::Action>>>
}

pub fn cache<C,M>(model: M) -> AnalysisCacheModel<C,M>
where
    M: Model,
    C: ShouldCache<State=M::State> + Send,
    <<M as Model>::Analyzer as GameAnalyzer>::State: PartialEq + Hash,
    <<M as Model>::Analyzer as GameAnalyzer>::Action: Clone,
{
    AnalysisCacheModel {
        model,
        cache: Arc::new(CHashMap::with_capacity(5_000_000)),
        phantom: PhantomData
    }
}

impl<C,M> Model for AnalysisCacheModel<C,M>
where
    M: Model,
    M::State: Send + Sync,
    M::Action: Clone + Send + Sync + 'static,
    C: ShouldCache<State=M::State> + Send + Sync,
    <<M as Model>::Analyzer as GameAnalyzer>::State: PartialEq + Hash + Send + Sync,
    <<M as Model>::Analyzer as GameAnalyzer>::Action: Clone + Send + Sync,
    <<M as Model>::Analyzer as GameAnalyzer>::Future: Unpin + 'static
{
    type State = M::State;
    type Analyzer = AnalysisCacheAnalyzer<C,M::Analyzer>;
    type Action = M::Action;

    fn get_model_info(&self) -> &ModelInfo {
        self.model.get_model_info()
    }

    fn train<I: Iterator<Item=PositionMetrics<Self::State,Self::Action>>>(&self, target_model_info: &ModelInfo, sample_metrics: I, options: &TrainOptions) -> Result<(), Error> {
        M::train(&self.model, target_model_info, sample_metrics, options)
    }

    fn get_game_state_analyzer(&self) -> Self::Analyzer {
        let analyzer = M::get_game_state_analyzer(&self.model);

        AnalysisCacheAnalyzer {
            analyzer,
            cache: self.cache.clone(),
            phantom: PhantomData
        }
    }
}

pub struct AnalysisCacheAnalyzer<C,Analyzer>
where
    Analyzer: GameAnalyzer,
    C: ShouldCache + Send,
    Analyzer::State: PartialEq + Hash,
    Analyzer::Action: Clone
{
    analyzer: Analyzer,
    phantom: PhantomData<C>,
    cache: Arc<CHashMap<Analyzer::State,GameStateAnalysis<Analyzer::Action>>>
}

impl<C,Analyzer> GameAnalyzer for AnalysisCacheAnalyzer<C,Analyzer>
where
    Analyzer: GameAnalyzer,
    C: ShouldCache<State=Analyzer::State> + Send,
    Analyzer::Future: Unpin + 'static,
    Analyzer::State: Clone + PartialEq + Hash + Send + Sync,
    Analyzer::Action: Clone + Send + Sync + 'static
{
    type State = Analyzer::State;
    type Action = Analyzer::Action;
    type Future = Box<dyn Future<Output=GameStateAnalysis<Self::Action>> + Unpin>;

    fn get_state_analysis(&self, game_state: &Self::State) -> Self::Future {
        let should_cache = C::should_cache(&game_state);
        let cache = &*self.cache;

        if should_cache {
            if let Some(analysis) = cache.get(&game_state) {
                let analysis = analysis.clone();
                let fut = (async { analysis }).boxed();
                return Box::new(fut);
            }
        }

        Box::new(Analyzer::get_state_analysis(&self.analyzer, game_state).map(move |analysis| {
            if should_cache {
                // cache.insert(game_state.clone(), analysis);
            }

            analysis
        }))
    }
}