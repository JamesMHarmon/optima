use serde::{Deserialize, Serialize};

pub trait PropagatedValue {
    fn value(&self) -> f32;
}

pub trait PropagatedGameLength {
    fn game_length(&self) -> f32;
}

#[derive(Default, Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct MovesLeftPropagatedValue {
    value: f32,
    game_length: f32,
    num_updates: usize,
}

impl MovesLeftPropagatedValue {
    pub fn new(value: f32, game_length: f32) -> Self {
        Self {
            value,
            game_length,
            num_updates: 0,
        }
    }

    pub fn value_mut(&mut self) -> &mut f32 {
        &mut self.value
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

impl PropagatedValue for MovesLeftPropagatedValue {
    fn value(&self) -> f32 {
        self.value
    }
}

impl PropagatedGameLength for MovesLeftPropagatedValue {
    fn game_length(&self) -> f32 {
        self.game_length
    }
}

impl Eq for MovesLeftPropagatedValue {}

impl Ord for MovesLeftPropagatedValue {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (self.game_length, self.value)
            .partial_cmp(&(other.game_length, other.value))
            .expect("Failed to compare")
    }
}

impl PartialOrd for MovesLeftPropagatedValue {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
