#![feature(toowned_clone_into)]
#![feature(test)]
#![allow(clippy::inconsistent_digit_grouping)]

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

use arimaa_engine::Action;
use constants::*;
use game_state::*;
use value::*;
