#![feature(is_some_with, test)]
#![allow(clippy::inconsistent_digit_grouping)]
#![allow(clippy::unusual_byte_groupings)]

mod board;
mod engine;

pub mod constants;
pub mod game_state;
pub mod mappings;
pub mod symmetries;
pub mod transposition_entry;
pub mod value;

pub use crate::engine::Engine;
pub use arimaa_engine::{convert_piece_to_letter, Action, Direction, Piece, Square};
pub use constants::*;
pub use game_state::*;
pub use mappings::*;
pub use symmetries::*;
pub use transposition_entry::*;
pub use value::*;

#[cfg(feature = "model")]
pub mod model;

#[cfg(feature = "model")]
pub mod model_factory;

#[cfg(feature = "model")]
pub use crate::model::*;

#[cfg(feature = "model")]
pub use crate::model_factory::*;
