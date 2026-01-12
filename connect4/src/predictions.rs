use engine::Value as ValueTrait;
use mcts::GameLength;
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

    pub fn value(&self) -> &Value {
        &self.value
    }

    pub fn game_length(&self) -> f32 {
        self.game_length
    }
}

impl ValueTrait for Predictions {
    fn get_value_for_player(&self, player: usize) -> f32 {
        self.value.get_value_for_player(player)
    }
}

impl GameLength for Predictions {
    fn game_length_score(&self) -> f32 {
        self.game_length
    }
}

impl std::fmt::Display for Predictions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Predictions(value: {}, game_length: {})",
            self.value(),
            self.game_length()
        )
    }
}
