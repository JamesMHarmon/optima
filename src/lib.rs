#![feature(async_await)]

pub mod engine;
pub mod mcts;
pub mod analytics;
pub mod analysis_cache;
pub mod game_state;
pub mod model;
pub mod node_metrics;
pub mod rng;
pub mod self_play;
pub mod self_play_persistance;
pub mod self_learn;

pub mod quoridor;
pub mod connect4;

mod bits;
mod futures;
