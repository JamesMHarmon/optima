#![allow(clippy::inconsistent_digit_grouping)]
#![allow(clippy::unusual_byte_groupings)]

mod board;
mod engine;

use board::*;

pub mod constants;
pub mod game_state;
pub mod mappings;
pub mod predictions;
pub mod selection_strategy;
pub mod symmetries;
pub mod time_strategy;
pub mod transposition_entry;
pub mod ugi;
pub mod value;

pub use crate::engine::Engine;
pub use arimaa_engine::{Action, Direction, Piece, Square, convert_piece_to_letter};
pub use constants::*;
pub use game_state::*;
pub use mappings::*;
pub use predictions::*;
pub use selection_strategy::*;
pub use symmetries::*;
pub use time_strategy::*;
pub use transposition_entry::*;
pub use ugi::*;
pub use value::*;

#[cfg(feature = "model")]
pub mod model;

#[cfg(feature = "model")]
pub mod model_factory;

#[cfg(feature = "model")]
pub use crate::model::*;

#[cfg(feature = "model")]
pub use crate::model_factory::*;
