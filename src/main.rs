extern crate quoridor;

use std::io::Write;
use std::fs::OpenOptions;
use std::env;

use pyo3::prelude::*;
use pyo3::types::PyList;

use std::time::{Instant};

use quoridor::self_play;
use quoridor::connect4::engine::{Engine as Connect4Engine};
use quoridor::analysis_cache::{AnalysisCache};
use quoridor::self_learn::{SelfLearn,SelfLearnOptions};

fn main() -> Result<(), &'static str>{
    SelfLearn::new("c4".to_string(), SelfLearnOptions {
        run_name: "Run_1".to_string(),
        number_of_games_per_net: 1000,
        moving_window_size: 10000,
        train_ratio: 0.9,
        train_batch_size: 512,
        epochs: 2,
        learning_rate: 0.001,
        policy_loss_weight: 1.0,
        value_loss_weight: 0.5,
        temperature: 1.0,
        visits: 800,
        cpuct: 4.0,
        number_of_filters: 128,
        number_of_residual_blocks: 5
    })
}

