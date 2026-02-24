use common::PlayerValue;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Value(arimaa_engine::Value);

impl Value {
    pub fn new(player_1: f32, player_2: f32) -> Self {
        Self(arimaa_engine::Value::new(player_1, player_2))
    }

    pub fn update_players_value(&mut self, player: usize, value: f32) {
        self.0.update_players_value(player, value);
    }
}

impl From<arimaa_engine::Value> for Value {
    fn from(value: arimaa_engine::Value) -> Self {
        Value(value)
    }
}

impl From<[f32; 2]> for Value {
    fn from(value: [f32; 2]) -> Self {
        Self(arimaa_engine::Value::new(value[0], value[1]))
    }
}

impl PlayerValue for Value {
    fn player_value(&self, player: usize) -> f32 {
        self.0.player_value(player)
    }
}
