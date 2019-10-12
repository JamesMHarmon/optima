use engine::game_state::GameState;
use failure::Error;
use serde::{Serialize,Deserialize};

use super::model_info::ModelInfo;
use super::position_metrics::PositionMetrics;
use super::analytics::GameAnalyzer;

pub trait Model {
    type State: GameState;
    type Action;
    type Value;
    type Analyzer: GameAnalyzer<Action=Self::Action,State=Self::State,Value=Self::Value> + Send;

    fn get_model_info(&self) -> &ModelInfo;
    fn train<I: Iterator<Item=PositionMetrics<Self::State,Self::Action,Self::Value>>>(&self, target_model_info: &ModelInfo, sample_metrics: I, options: &TrainOptions) -> Result<(), Error>;
    fn get_game_state_analyzer(&self) -> Self::Analyzer;
}

pub trait ModelFactory
{
    type M: Model;
    type O;

    fn create(&self, model_info: &ModelInfo, model_options: &Self::O) -> Self::M;
    fn get(&self, model_info: &ModelInfo) -> Self::M;
    fn get_latest(&self, model_info: &ModelInfo) -> Result<ModelInfo,Error>;
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ModelOptions {
    pub number_of_filters: usize,
    pub number_of_residual_blocks: usize
}

pub struct TrainOptions {
    pub train_ratio: f32,
    pub train_batch_size: usize,
    pub epochs: usize,
    pub learning_rate: f32,
    pub policy_loss_weight: f32,
    pub value_loss_weight: f32
}
