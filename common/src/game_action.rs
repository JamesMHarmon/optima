use serde::Serialize;
use serde::de::DeserializeOwned;
use std::fmt::Debug;
use std::hash::Hash;

/// Supertrait capturing the standard bounds required on a game action.
///
/// Every type that satisfies the individual bounds automatically implements
/// this trait via the blanket impl below, so no manual implementation is
/// needed at the concrete action types.
pub trait GameAction:
    Clone + Eq + Hash + DeserializeOwned + Serialize + Debug + Unpin + Send + Sync
{
}

impl<T> GameAction for T where
    T: Clone + Eq + Hash + DeserializeOwned + Serialize + Debug + Unpin + Send + Sync
{
}
