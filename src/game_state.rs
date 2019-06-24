use std::hash::Hash;

pub trait GameState: Hash + Eq + Clone {
    fn initial() -> Self;
}
