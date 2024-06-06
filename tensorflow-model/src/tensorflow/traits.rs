use std::collections::HashMap;

use half::f16;
use model::{ActionWithPolicy, GameStateAnalysis, NodeMetrics};
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

pub trait InputMap {
    type State;

    fn game_state_to_input(
        &self,
        game_state: &Self::State,
        inputs: HashMap<&str, &mut [f16]>,
        mode: Mode,
    );
}

pub trait PolicyMap {
    type State;
    type Action;
    type Predictions;

    // fn policy_metrics_to_expected_output(
    //     &self,
    //     game_state: &Self::State,
    //     policy: &NodeMetrics<Self::Action, Self::Predictions>,
    // ) -> Vec<f32>;

    // fn policy_to_valid_actions(
    //     &self,
    //     game_state: &Self::State,
    //     policy_scores: &[f16],
    // ) -> Vec<ActionWithPolicy<Self::Action>>;
}

pub trait PredictionsMap<S, P> {
    type State;
    type Predictions;

    // fn from_output(&self, game_state: &Self::State, prediction_output: Option<HashMap<String, &[f16]>>) -> Self::Predictions;

    // fn to_output(&self, game_state: &Self::State, predictions: &Self::Predictions) -> HashMap<String, Vec<f16>>;
}

pub trait TranspositionMap {
    type State;
    type TranspositionEntry;
    type GameStateAnalysis;

    fn map_output_to_transposition_entry(
        &self,
        game_state: &Self::State,
        outputs: HashMap<String, &[f16]>,
    ) -> Self::TranspositionEntry;

    fn map_transposition_entry_to_analysis(
        &self,
        game_state: &Self::State,
        transposition_entry: &Self::TranspositionEntry,
    ) -> Self::GameStateAnalysis;

    fn get_transposition_key(&self, game_state: &Self::State) -> u64;
}
