use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Value(arimaa_engine::Value);

impl From<arimaa_engine::Value> for Value {
    fn from(value: arimaa_engine::Value) -> Self {
        Value(value)
    }
}

impl engine::value::Value for Value {
    fn get_value_for_player(&self, player: usize) -> f32 {
        self.0 .0[player - 1]
    }
}
