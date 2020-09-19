#![allow(clippy::inconsistent_digit_grouping)]

#[macro_use]
mod bits;
mod board;
mod zobrist;
mod zobrist_values;

pub mod action;
pub mod constants;
pub mod engine;
pub mod model;
pub mod value;
