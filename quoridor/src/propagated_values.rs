use common::{PropagatedGameLength, PropagatedValue};
use serde::{Deserialize, Serialize};

#[derive(Default, Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct QuoridorPropagatedValue {
    value: f32,
    game_length: f32,
    victory_margin: f32,
    num_updates: usize,
}

impl QuoridorPropagatedValue {
    pub fn new(value: f32, victory_margin: f32, game_length: f32) -> Self {
        Self {
            value,
            game_length,
            victory_margin,
            num_updates: 0,
        }
    }

    pub fn value_mut(&mut self) -> &mut f32 {
        &mut self.value
    }

    pub fn victory_margin(&self) -> f32 {
        self.victory_margin
    }

    pub fn victory_margin_mut(&mut self) -> &mut f32 {
        &mut self.victory_margin
    }

    pub fn game_length_mut(&mut self) -> &mut f32 {
        &mut self.game_length
    }

    pub fn num_updates(&self) -> usize {
        self.num_updates
    }

    pub fn increment_num_updates(&mut self) {
        self.num_updates += 1;
    }
}

impl PropagatedValue for QuoridorPropagatedValue {
    fn value(&self) -> f32 {
        self.value
    }
}

impl PropagatedGameLength for QuoridorPropagatedValue {
    fn game_length(&self) -> f32 {
        self.game_length
    }
}

impl Eq for QuoridorPropagatedValue {}

impl Ord for QuoridorPropagatedValue {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (self.game_length, self.victory_margin, self.value)
            .partial_cmp(&(other.game_length, other.victory_margin, other.value))
            .expect("Failed to compare")
    }
}

impl PartialOrd for QuoridorPropagatedValue {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
