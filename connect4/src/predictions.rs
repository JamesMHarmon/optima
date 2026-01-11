use engine::Value as ValueTrait;

use super::Value;

#[derive(Clone)]
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
