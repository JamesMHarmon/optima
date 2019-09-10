use engine::game_state::GameState;
use failure::Error;

use super::model_info::ModelInfo;
use super::position_metrics::PositionMetrics;
use super::analytics::GameAnalyzer;

pub trait Model {
    type State: GameState;
    type Analyzer: GameAnalyzer<Action=Self::Action,State=Self::State> + Send;
    type Action;

    fn get_model_info(&self) -> &ModelInfo;
    fn train<I: Iterator<Item=PositionMetrics<Self::State,Self::Action>>>(&self, target_model_info: &ModelInfo, sample_metrics: I, options: &TrainOptions) -> Result<(), Error>;
    fn get_game_state_analyzer(&self) -> Self::Analyzer;
}

pub trait ModelFactory
{
    type M: Model;

    fn create(&self, model_info: &ModelInfo, num_filters: usize, num_blocks: usize) -> Self::M;
    fn get(&self, model_info: &ModelInfo) -> Self::M;
    fn get_latest(&self, model_info: &ModelInfo) -> Result<ModelInfo,Error>;
}

pub struct TrainOptions {
    pub train_ratio: f64,
    pub train_batch_size: usize,
    pub epochs: usize,
    pub learning_rate: f64,
    pub policy_loss_weight: f64,
    pub value_loss_weight: f64
}
