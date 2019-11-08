#![feature(toowned_clone_into)]
#![feature(vec_remove_item)]
#![feature(test)]

#[macro_use]
mod bits;
mod board;
mod place_model;
mod play_model;
mod zobrist;
mod zobrist_values;

pub mod action;
pub mod engine;
pub mod model;
pub mod constants;
pub mod value;
