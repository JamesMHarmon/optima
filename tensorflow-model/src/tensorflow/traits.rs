
use half::f16;
use model::{NodeMetrics, ActionWithPolicy, GameStateAnalysis};
use serde::{Deserialize, Serialize};

use super::Mode;

#[derive(Serialize, Deserialize)]
pub struct TensorflowModelOptions {
    pub num_filters: usize,
    pub num_blocks: usize,
    pub channel_height: usize,
    pub channel_width: usize,
    pub channels: usize,
    pub output_size: usize,
    pub moves_left_size: usize,
}

pub trait Dimension {
    fn dimensions(&self) -> [u64; 3];
}

pub trait InputMap<S> {
    fn game_state_to_input(&self, game_state: &S, mode: Mode) -> Vec<half::f16>;
}

pub trait PolicyMap<S, A> {
    fn policy_metrics_to_expected_output(
        &self,
        game_state: &S,
        policy: &NodeMetrics<A>,
    ) -> Vec<f32>;

    fn policy_to_valid_actions(
        &self,
        game_state: &S,
        policy_scores: &[f16],
    ) -> Vec<ActionWithPolicy<A>>;
}

pub trait ValueMap<S, V> {
    fn map_value_to_value_output(&self, game_state: &S, value: &V) -> f32;

    fn map_value_output_to_value(&self, game_state: &S, value_output: f32) -> V;
}

pub trait TranspositionMap<S, A, V, Te> {
    fn map_output_to_transposition_entry<I: Iterator<Item = f16>>(
        &self,
        game_state: &S,
        policy_scores: I,
        value: f16,
        moves_left: f32,
    ) -> Te;

    fn map_transposition_entry_to_analysis(
        &self,
        game_state: &S,
        transposition_entry: &Te,
    ) -> GameStateAnalysis<A, V>;

    fn get_transposition_key(&self, game_state: &S) -> u64;
}
