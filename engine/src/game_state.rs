use std::fmt::Debug;

pub trait GameState: Clone + Debug {
    fn initial() -> Self;
}
