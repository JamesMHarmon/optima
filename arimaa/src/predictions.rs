use std::fmt::{Display, Formatter};

use common::GameLength;
use serde::{Deserialize, Serialize};

use super::Value;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Predictions {
    value: Value,
    game_length: f32,
}

impl Predictions {
    pub fn new(value: Value, game_length: f32) -> Self {
        Self { value, game_length }
    }

    pub fn value(&self) -> &Value {
        &self.value
    }

    pub fn game_length(&self) -> f32 {
        self.game_length
    }
}

impl Display for Predictions {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "value: {:?}, game_length: {}",
            self.value, self.game_length
        )
    }
}

impl engine::Value for Predictions {
    fn get_value_for_player(&self, player: usize) -> f32 {
        self.value().get_value_for_player(player)
    }
}

impl GameLength for Predictions {
    fn game_length_score(&self) -> f32 {
        self.game_length
    }
}
