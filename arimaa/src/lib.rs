#![feature(test)]
#![allow(clippy::inconsistent_digit_grouping)]
#![allow(clippy::unusual_byte_groupings)]

mod board;
mod engine;

pub mod constants;
pub mod game_state;
pub mod place_mappings;
pub mod play_mappings;
pub mod symmetries;
pub mod transposition_entry;
pub mod value;

pub use crate::engine::Engine;
pub use arimaa_engine::{convert_piece_to_letter, Action, Direction, Piece, Square};
pub use constants::*;
pub use game_state::*;
pub use place_mappings::{Mapper as PlaceMapper, *};
pub use play_mappings::{Mapper as PlayMapper, *};
pub use symmetries::*;
pub use transposition_entry::*;
pub use value::*;

#[cfg(feature = "model")]
pub mod model;
#[cfg(feature = "model")]
pub mod place_model;
#[cfg(feature = "model")]
pub mod play_model;

#[cfg(feature = "model")]
pub use crate::model::*;

#[cfg(feature = "model")]
pub use crate::place_model::*;

#[cfg(feature = "model")]
pub use crate::play_model::*;
