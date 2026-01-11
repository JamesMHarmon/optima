use super::OUTPUT_SIZE;

use half::f16;

pub struct TranspositionEntry {
    policy_metrics: [f16; OUTPUT_SIZE],
    value: f16,
    game_length: f32,
}

impl TranspositionEntry {
    pub fn new(policy_metrics: [f16; OUTPUT_SIZE], value: f16, game_length: f32) -> Self {
        Self {
            policy_metrics,
            value,
            game_length,
        }
    }

    pub fn policy_metrics(&self) -> &[f16; OUTPUT_SIZE] {
        &self.policy_metrics
    }

    pub fn value(&self) -> f16 {
        self.value
    }

    pub fn game_length(&self) -> f32 {
        self.game_length
    }
}
