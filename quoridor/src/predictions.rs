use super::Value;
use common::GameLength;
use common::PlayerValue;
use common::VictoryMargin;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Predictions {
    value: Value,
    victory_margin: f32,
    game_length: f32,
}

impl Predictions {
    pub fn new(value: Value, victory_margin: f32, game_length: f32) -> Self {
        Self {
            value,
            victory_margin,
            game_length,
        }
    }

    pub fn value(&self) -> &Value {
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

impl VictoryMargin for Predictions {
    fn victory_margin(&self) -> f32 {
        self.victory_margin
    }
}

impl std::fmt::Display for Predictions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Predictions(value: {}, victory_margin: {}, game_length: {})",
            self.value, self.victory_margin, self.game_length
        )
    }
}
