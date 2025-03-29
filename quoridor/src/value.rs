use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Value(pub [f32; 2]);

impl Value {
    pub fn new(values: [f32; 2]) -> Self {
        Self(values)
    }

    pub fn update_players_value(&mut self, player: usize, value: f32) {
        self.0[player - 1] = value
    }
}

impl engine::value::Value for Value {
    fn get_value_for_player(&self, player: usize) -> f32 {
        self.0[player - 1]
    }
}
