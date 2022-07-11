use half::f16;

pub struct TranspositionEntry<P> {
    policy_metrics: P,
    moves_left: f32,
    value: f16,
}

impl<P> TranspositionEntry<P> {
    pub fn new(policy_metrics: P, value: f16, moves_left: f32) -> Self {
        Self {
            policy_metrics,
            moves_left,
            value,
        }
    }

    pub fn policy_metrics(&self) -> &P {
        &self.policy_metrics
    }

    pub fn moves_left(&self) -> f32 {
        self.moves_left
    }

    pub fn value(&self) -> f16 {
        self.value
    }
}
