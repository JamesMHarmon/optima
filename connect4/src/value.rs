use common::PlayerValue;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Value([f32; 2]);

impl Value {
    pub fn new(p1_value: f32, p2_value: f32) -> Self {
        Self([p1_value, p2_value])
    }

    pub fn update_players_value(&mut self, player: usize, value: f32) {
        self.0[player - 1] = value;
    }
}

impl PlayerValue for Value {
    fn player_value(&self, player: usize) -> f32 {
        self.0[player - 1]
    }
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Value({}, {})", self.0[0], self.0[1])
    }
}
