extern crate quoridor;

use rand::RngCore;
use pyo3::prelude::*;
use pyo3::types::PyList;
use std::env;

use rand::prelude::{SeedableRng, StdRng};
use std::time::{Instant};
use uuid::Uuid;

use quoridor::mcts::{DirichletOptions,MCTS,MCTSOptions};
use quoridor::connect4::engine::{GameState, Engine as Connect4Engine};
use quoridor::engine::GameEngine;
use quoridor::analysis_cache::{AnalysisCache};

fn main() {
    set_path();

    let game_engine = Connect4Engine::new();
    let game_state = GameState::new();
    let analysis_cache = AnalysisCache::new();
    let seedable_rng = create_rng();

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
        println!("Metrics: {:?}", metrics);
        println!("Action: {:?}", action);
    }

    let time = now.elapsed().as_millis();

    println!("Result: {}", game_engine.is_terminal_state(&state).unwrap());
    println!("Last Player: {}", if state.p1_turn_to_move { "P2" } else { "P1" });
    println!("TIME: {}",time);
}

fn create_rng() -> impl RngCore {
    let uuid = Uuid::new_v4();
    let uuid_bytes: &[u8; 16] = uuid.as_bytes();
    let mut seed = [0; 32];
    seed[..16].clone_from_slice(uuid_bytes);
    seed[16..32].clone_from_slice(uuid_bytes);

    println!("uuid: {}", uuid);

    let seedable_rng: StdRng = SeedableRng::from_seed(seed);

    seedable_rng
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
