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

fn main() -> Result<(), &'static str>{
    set_python_paths();
    create_model();

    let game_engine = Connect4Engine::new();
    let mut file = OpenOptions::new()
        .append(true)
        .create(true)
        .open("results.txt")
        .expect("Couldn't open or create the results.txt file");

    loop {
        let mut analysis_cache = AnalysisCache::new();

        let now = Instant::now();
        let self_play_metrics = self_play::self_play(&game_engine, &mut analysis_cache)?;
        let time = now.elapsed().as_millis();

        let serialized = serde_json::to_string(&self_play_metrics).expect("Failed to serialize results");

        writeln!(file, "{}", serialized).expect("File to write to results.txt.");

        println!("{:?}", self_play_metrics);
        println!("TIME: {}",time);
    }
}

// @TODO: Improve error handling
fn set_python_paths() {
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

fn create_model() {
    let gil = Python::acquire_gil();
    let py = gil.python();
    let c4 = py.import("c4_model").unwrap();

    c4.call("create_model", (), None).unwrap();
}
