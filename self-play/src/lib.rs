#![feature(is_some_with)]

#[cfg(feature = "all")]
pub mod options;
#[cfg(feature = "all")]
pub mod play_self_one;
#[cfg(feature = "all")]
pub mod self_play;
#[cfg(feature = "all")]
pub mod self_play_persistance;

#[cfg(feature = "all")]
pub use crate::self_play::*;
#[cfg(feature = "all")]
pub use options::*;
#[cfg(feature = "all")]
pub use play_self_one::*;
#[cfg(feature = "all")]
pub use self_play_persistance::*;

pub mod self_play_metrics;
pub use self_play_metrics::*;
