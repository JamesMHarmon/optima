use std::collections::HashMap;

use half::f16;
use model::{GameStateAnalysis, NodeMetrics};
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

    fn game_state_to_input(&self, game_state: &Self::State, inputs: &mut [f16], mode: Mode);
}

pub trait PredictionsMap {
    type State;
    type Action;
    type Predictions;
    type PropagatedValues;

    fn to_output(
        &self,
        game_state: &Self::State,
        targets: Self::Predictions,
        node_metrics: &NodeMetrics<Self::Action, Self::Predictions, Self::PropagatedValues>,
    ) -> HashMap<String, Vec<f32>>;
}

pub trait TranspositionMap {
    type State;
    type Action;
    type TranspositionEntry;
    type Predictions;

    fn map_output_to_transposition_entry(
        &self,
        game_state: &Self::State,
        outputs: HashMap<String, &[f16]>,
    ) -> Self::TranspositionEntry;

    fn map_transposition_entry_to_analysis(
        &self,
        game_state: &Self::State,
        transposition_entry: &Self::TranspositionEntry,
    ) -> GameStateAnalysis<Self::Action, Self::Predictions>;

    fn get_transposition_key(&self, game_state: &Self::State) -> u64;
}
