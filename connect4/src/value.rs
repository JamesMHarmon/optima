use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Value(pub [f32; 2]);

impl engine::value::Value for Value {
    fn get_value_for_player(&self, player: usize) -> f32 {
        self.0[player - 1]
    }
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Value({}, {})", self.0[0], self.0[1])
    }
}
