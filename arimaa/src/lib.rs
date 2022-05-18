#![feature(toowned_clone_into)]
#![feature(test)]
#![allow(clippy::inconsistent_digit_grouping)]
#![allow(clippy::unusual_byte_groupings)]

mod board;
mod engine;
mod place_model;

pub mod constants;
pub mod game_state;
pub mod model;
pub mod play_model;
pub mod symmetries;
pub mod value;

pub use crate::engine::Engine;
pub use crate::model::*;

pub use arimaa_engine::{Action, Direction, Piece, Square, convert_piece_to_letter};
pub use constants::*;
pub use game_state::*;
pub use value::*;
