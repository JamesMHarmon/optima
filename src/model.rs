use super::self_play::SelfPlaySample;
use super::game_state::GameState;

pub trait Model {
    type State: GameState;
    type Action;

    fn get_name(&self) -> &str;
    fn train(&self, target_name: &str, sample_metrics: &Vec<SelfPlaySample<Self::State, Self::Action>>, options: &TrainOptions) -> Self;
}

pub trait ModelFactory
{
    type M: Model;

    fn create(&self, name: &str, num_filters: usize, num_blocks: usize) -> Self::M;
    fn get_latest(&self, name: &str) -> Self::M;
}

pub struct TrainOptions {
    pub train_ratio: f64,
    pub train_batch_size: usize,
    pub epochs: usize,
    pub learning_rate: f64,
    pub policy_loss_weight: f64,
    pub value_loss_weight: f64
}
