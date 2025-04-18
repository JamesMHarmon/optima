#![allow(clippy::inconsistent_digit_grouping)]
#![allow(clippy::unusual_byte_groupings)]

#[macro_use]
mod bits;
mod game_state_test;
mod serde;
mod zobrist;
mod zobrist_values;

pub mod action;
pub mod constants;
pub mod coordinate;
pub mod display;
pub mod engine;
pub mod game_state;
pub mod mappings;
pub mod predictions;
pub mod propagated_values;
pub mod selection_strategy;
pub mod symmetries;
pub mod transposition_entry;
pub mod value;

pub use action::*;
pub use constants::*;
pub use coordinate::*;
pub use engine::Engine;
pub use game_state::*;
pub use mappings::*;
pub use predictions::*;
pub use propagated_values::*;
pub use selection_strategy::*;
pub use symmetries::*;
pub use transposition_entry::*;
pub use value::*;

use zobrist::*;

#[cfg(feature = "model")]
pub mod model;

#[cfg(feature = "model")]
pub mod model_factory;

#[cfg(feature = "model")]
pub mod ugi;

#[cfg(feature = "model")]
pub use crate::model::*;

#[cfg(feature = "model")]
pub use crate::model_factory::*;

#[cfg(feature = "model")]
pub use ugi::*;
