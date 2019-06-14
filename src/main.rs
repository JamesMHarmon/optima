extern crate quoridor;

use pyo3::prelude::*;
use pyo3::types::PyList;
use std::env;

use rand::prelude::{SeedableRng, StdRng};
use std::time::{Instant};

use quoridor::mcts::{DirichletOptions,MCTS,MCTSOptions};
use quoridor::quoridor::engine::{GameState, Engine as QuoridorEngine};

fn main() -> PyResult<()> {
    let game_engine = QuoridorEngine {};
    let game_state = GameState::new();
    let seed: [u8; 32] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32];
    let seedable_rng: StdRng = SeedableRng::from_seed(seed);
    let mut mcts = MCTS::new(
        game_state,
        &game_engine,
        MCTSOptions::new(
            Some(DirichletOptions {
                alpha: 0.3,
                epsilon: 0.25
            }),
            &|_,_| { 4.0 },
            &|_| { 1.0 },
            seedable_rng,
        )
    );


    let now = Instant::now();
    let res = mcts.get_next_action(1).unwrap();
    let time = now.elapsed().as_millis();

    println!("TIME: {}",time);

    println!("{:?}", res);
    // println!("{:?}", mcts.get_next_action(1));
    // println!("{:?}", mcts.get_next_action(800));
    // println!("{:?}", mcts.get_next_action(800));

    let gil = Python::acquire_gil();
    let py = gil.python();

    let sys = py.import("sys")?;
    let os = py.import("os")?;
    let version: String = sys.get("version")?.extract()?;
    println!("Version: {}", version);
    let path = sys.get("path")?.downcast_ref::<PyList>()?;
    let cwd: String = os.call("getcwd", (), None)?.extract()?;
    println!("CWD: {}", cwd);

    let current_dir_result = env::current_dir()?;
    let env_path = current_dir_result.to_str().unwrap();
    println!("Env Path: {}", env_path);

    path.call_method("append", (env_path.to_owned(), ), None)?;
    path.call_method("append", ("/anaconda3/lib/python3.6".to_owned(), ), None)?;
    path.call_method("append", ("/anaconda3/lib/python3.6/lib-dynload".to_owned(), ), None)?;
    path.call_method("append", ("/anaconda3/lib/python3.6/site-packages", ), None)?;

    let module_name = "c4_model";
    println!("Loading: {}", module_name);
    let test = py.import(module_name)?;
    println!("Successfully Loaded: {}", module_name);
    
    let test_result: String = test.call("getBestMove", ("121213",), None)?.extract()?;
    println!("Best Move: {}", test_result);

    Ok(())
}