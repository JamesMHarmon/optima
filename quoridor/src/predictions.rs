use super::{Value, VictoryMargin};
use common::GameLength;
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

    pub fn victory_margin(&self) -> f32 {
        self.victory_margin
    }

    pub fn game_length(&self) -> f32 {
        self.game_length
    }
}

impl engine::Value for Predictions {
    fn get_value_for_player(&self, player: usize) -> f32 {
        self.value().get_value_for_player(player)
    }
}

impl GameLength for Predictions {
    fn game_length_score(&self) -> f32 {
        self.game_length()
    }
}

impl VictoryMargin for Predictions {
    fn victory_margin_score(&self) -> f32 {
        self.victory_margin()
    }
}

impl std::fmt::Display for Predictions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Predictions(value: {}, victory_margin: {}, game_length: {})",
            self.value(),
            self.victory_margin(),
            self.game_length()
        )
    }
}
