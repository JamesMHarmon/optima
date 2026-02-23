use common::{GameLength, PlayerValue};
use serde::{Deserialize, Serialize};

use super::Value;

#[derive(Clone, Serialize, Deserialize)]
pub struct Predictions {
    value: Value,
    game_length: f32,
}

impl Predictions {
    pub fn new(value: Value, game_length: f32) -> Self {
        Self { value, game_length }
    }

    pub(super) fn value(&self) -> &Value {
        &self.value
    }
}

impl PlayerValue for Predictions {
    fn player_value(&self, player: usize) -> f32 {
        self.value.player_value(player)
    }
}

impl GameLength for Predictions {
    fn game_length(&self) -> f32 {
        self.game_length
    }
}

impl std::fmt::Display for Predictions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Predictions(value: {}, game_length: {})",
            self.value, self.game_length
        )
    }
}
