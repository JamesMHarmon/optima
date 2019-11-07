use std::future::Future;
use model::analysis_cache_queue::{AnalysisCacheQueueModel,AnalysisCacheAnalyzer};
use model::position_metrics::PositionMetrics;
use model::model::TrainOptions;
use model::model::ModelOptions;
use model::model_info::ModelInfo;
use model::tensorflow::model::TensorflowModel;
use model::tensorflow::get_latest_model_info::get_latest_model_info;
use model::tensorflow::model::*;
use futures::future::Either;
use super::value::Value;
use super::action::Action;
use super::engine::Engine;
use super::engine::GameState;
use super::place_model::{ModelFactory as PlaceModelFactory, Mapper as PlaceMapper};
use super::play_model::{ModelFactory as PlayModelFactory, Mapper as PlayMapper};

use failure::Error;

pub struct ModelFactory {
    play_model_factory: PlayModelFactory,
    place_model_factory: PlaceModelFactory
}

impl ModelFactory {
    pub fn new() -> Self {
        ModelFactory {
            play_model_factory: PlayModelFactory::new(),
            place_model_factory: PlaceModelFactory::new()
        }
    }
}

impl model::model::ModelFactory for ModelFactory {
    type M = Model;
    type O = ModelOptions;

    fn create(&self, model_info: &ModelInfo, options: &Self::O) -> Self::M
    {
        let play_model = self.play_model_factory.create(model_info, options);
        let place_model_info = map_to_place_model_info(model_info);
        let place_model = self.place_model_factory.create(&place_model_info, options);

        Model {
            play_model,
            place_model
        }
    }

    fn get(&self, model_info: &ModelInfo) -> Self::M {
        let play_model = self.play_model_factory.get(model_info);
        let place_model_info = map_to_place_model_info(model_info);
        let place_model = self.place_model_factory.get(&place_model_info);

        Model {
            play_model,
            place_model
        }
    }

    fn get_latest(&self, model_info: &ModelInfo) -> Result<ModelInfo, Error> {
        Ok(get_latest_model_info(model_info)?)
    }
}

pub struct Model {
    play_model: AnalysisCacheQueueModel<TensorflowModel<Engine,PlayMapper>>,
    place_model: TensorflowModel<Engine,PlaceMapper>
}

impl model::model::Model for Model {
    type State = GameState;
    type Action = Action;
    type Value = Value;
    type Analyzer = Analyzer;
    
    fn get_model_info(&self) -> &ModelInfo {
        self.play_model.get_model_info()
    }
    
    fn train<I>(&self, target_model_info: &ModelInfo, sample_metrics: I, options: &TrainOptions) -> Result<(), Error>
    where
        I: Iterator<Item=PositionMetrics<Self::State,Self::Action,Self::Value>>
    {
        let mut place_samples = Vec::new();

        let play_sample_iter = sample_metrics.filter_map(|sample_metric| {
            if sample_metric.game_state.is_play_phase() {
                Some(sample_metric)
            } else {
                place_samples.push(sample_metric);
                None
            }
        });

        self.play_model.train(target_model_info, play_sample_iter, options)?;

        let place_model_info = map_to_place_model_info(target_model_info);
        self.place_model.train(&place_model_info, place_samples.into_iter(), options)
    }
    
    fn get_game_state_analyzer(&self) -> Self::Analyzer {
        Analyzer {
            play_analyzer: self.play_model.get_game_state_analyzer(),
            place_analyzer: self.place_model.get_game_state_analyzer()
        }
    }
}

pub struct Analyzer {
    play_analyzer: AnalysisCacheAnalyzer<GameAnalyzer<Engine,PlayMapper>>,
    place_analyzer: GameAnalyzer<Engine,PlaceMapper>
}

impl model::analytics::GameAnalyzer for Analyzer {
    type Future = Either<
        std::pin::Pin<std::boxed::Box<(dyn Future<Output = model::analytics::GameStateAnalysis<Action, Value>> + Send + 'static)>>,
        GameStateAnalysisFuture<Self::State,Engine,PlaceMapper>
    >;
    type Action = Action;
    type State = GameState;
    type Value = Value;

    fn get_state_analysis(&self, game_state: &Self::State) -> Self::Future {
        if game_state.is_play_phase() {
            Either::Left(self.play_analyzer.get_state_analysis(game_state))
        } else {
            Either::Right(self.place_analyzer.get_state_analysis(game_state))
        }
    }
}

fn map_to_place_model_info(model_info: &ModelInfo) -> ModelInfo {
    ModelInfo::new(
        model_info.get_game_name().to_owned(),
        model_info.get_run_name().to_owned() + "-place",
        model_info.get_model_num()
    )
}
