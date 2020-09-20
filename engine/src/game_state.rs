use std::fmt::Debug;
use std::hash::Hash;

pub trait GameState: Hash + Clone + Debug {
    fn initial() -> Self;
}
