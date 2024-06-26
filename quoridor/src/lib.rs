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
pub mod symmetries;
pub mod transposition_entry;
pub mod ugi;
pub mod value;

pub use action::*;
pub use constants::*;
pub use coordinate::*;
pub use engine::Engine;
pub use game_state::*;
pub use mappings::*;
pub use symmetries::*;
pub use transposition_entry::*;
pub use ugi::*;
pub use value::*;

use zobrist::*;

#[cfg(feature = "model")]
pub mod model;

#[cfg(feature = "model")]
pub mod model_factory;

#[cfg(feature = "model")]
pub use crate::model::*;

#[cfg(feature = "model")]
pub use crate::model_factory::*;
