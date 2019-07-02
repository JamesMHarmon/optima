use std::fmt::Debug;
use std::hash::Hash;

pub trait GameState: Hash + Eq + Clone + Debug {
    fn initial() -> Self;
}
