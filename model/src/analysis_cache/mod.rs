use std::marker::PhantomData;
use super::analytics::GameAnalyzer;
use super::model::TrainOptions;
use super::position_metrics::PositionMetrics;
use super::model::Model;
use super::model_info::ModelInfo;

use failure::Error;

pub trait ShouldCache {
    type State;

    fn should_cache(game_state: &Self::State) -> bool;
}

pub struct AnalysisCacheModel<C,M>
where
    M: Model,
    C: ShouldCache<State=M::State> + Send
{
    model: M,
    phantom: PhantomData<C>
}

pub fn cache<C,M>(model: M) -> AnalysisCacheModel<C,M>
where
    M: Model,
    C: ShouldCache<State=M::State> + Send
{
    AnalysisCacheModel {
        model,
        phantom: PhantomData
    }
}

impl<C,M> Model for AnalysisCacheModel<C,M>
where
    M: Model,
    C: ShouldCache<State=M::State> + Send + Sync
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
            phantom: PhantomData
        }
    }
}

pub struct AnalysisCacheAnalyzer<C,A>
where
    A: GameAnalyzer,
    C: ShouldCache + Send
{
    analyzer: A,
    phantom: PhantomData<C>
}

// use futures::future::FutureExt;
// use std::future::Future;
// use futures::future::map::Map;

impl<C,A> GameAnalyzer for AnalysisCacheAnalyzer<C,A>
where
    A: GameAnalyzer,
    C: ShouldCache<State=A::State> + Send
{
    type State = A::State;
    type Action = A::Action;
    type Future = A::Future;

    fn get_state_analysis(&self, game_state: &Self::State) -> A::Future {
        let should_cache = C::should_cache(&game_state);

        if should_cache {

        }

        A::get_state_analysis(&self.analyzer, game_state)
    }
}