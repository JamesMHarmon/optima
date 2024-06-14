pub trait PropagatedValue {
    fn value(&self) -> f32;
}

pub trait PropagatedGameLength {
    fn game_length(&self) -> f32;
}
