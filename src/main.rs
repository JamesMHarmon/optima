extern crate quoridor;

use pyo3::prelude::*;
use pyo3::types::PyList;
use std::env;

use rand::prelude::{SeedableRng, StdRng};
use std::time::{Instant};

use quoridor::mcts::{DirichletOptions,MCTS,MCTSOptions};
use quoridor::connect4::engine::{GameState, Engine as Connect4Engine};
use quoridor::engine::GameEngine;
use quoridor::analysis_cache::{AnalysisCache};

fn main() {
    set_path();

    let game_engine = Connect4Engine::new();
    let game_state = GameState::new();
    let analysis_cache = AnalysisCache::new();
    // @TODO: Convert seed to use a guid
    let seed: [u8; 32] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32];
    let seedable_rng: StdRng = SeedableRng::from_seed(seed);
    let mut mcts = MCTS::new(
        game_state,
        &game_engine,
        analysis_cache,
        MCTSOptions::new(
            Some(DirichletOptions {
                alpha: 0.3,
                epsilon: 0.25
            }),
            &|_,_| 4.0,
            &|_| 1.0,
            seedable_rng,
        )
    );

    let now = Instant::now();

    let mut state: GameState = GameState::new();

    while game_engine.is_terminal_state(&state) == None {
        let search_result = mcts.search(800).unwrap();
        let action = search_result.0;
        let metrics = mcts.get_root_node_metrics();
        mcts.advance_to_action(&action).unwrap();
        state = game_engine.take_action(&state, &action);
        println!("Action: {:?}", action);
        println!("Metrics: {:?}", metrics);
    }

    let time = now.elapsed().as_millis();

    println!("TIME: {}",time);
}

// @TODO: Improve error handling
fn set_path() {
    let gil = Python::acquire_gil();
    let py = gil.python();

    let current_dir_result = env::current_dir().unwrap();
    let env_path = current_dir_result.to_str().ok_or("Path not valid").unwrap();
    println!("Env Path: {}", env_path);

    let sys = py.import("sys").unwrap();
    let path = sys.get("path").unwrap().downcast_ref::<PyList>().unwrap();

    path.call_method("append", (env_path.to_owned(), ), None).unwrap();
    path.call_method("append", ("/anaconda3/lib/python3.6".to_owned(), ), None).unwrap();
    path.call_method("append", ("/anaconda3/lib/python3.6/lib-dynload".to_owned(), ), None).unwrap();
    path.call_method("append", ("/anaconda3/lib/python3.6/site-packages", ), None).unwrap();
}
