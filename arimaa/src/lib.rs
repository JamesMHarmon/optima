#![feature(test)]
#![allow(clippy::inconsistent_digit_grouping)]
#![allow(clippy::unusual_byte_groupings)]

mod board;
mod engine;

pub mod constants;
pub mod game_state;
pub mod model;
pub mod place_model;
pub mod play_model;
pub mod symmetries;
pub mod value;

pub use crate::engine::Engine;
pub use crate::model::*;

pub use arimaa_engine::{convert_piece_to_letter, Action, Direction, Piece, Square};
pub use constants::*;
pub use game_state::*;
pub use symmetries::*;
pub use value::*;
