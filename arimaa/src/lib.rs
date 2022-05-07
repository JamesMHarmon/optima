#![feature(toowned_clone_into)]
#![feature(test)]
#![allow(clippy::inconsistent_digit_grouping)]
#![allow(clippy::unusual_byte_groupings)]

#[macro_use]
mod bits;
mod board;
mod place_model;
mod zobrist;
mod zobrist_values;

pub mod action;
pub mod constants;
pub mod display;
pub mod engine;
pub mod model;
pub mod play_model;
pub mod symmetries;
pub mod value;
