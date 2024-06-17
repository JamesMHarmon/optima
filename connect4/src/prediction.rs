use super::Value;

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
