#![feature(async_await)]

mod self_evaluate;

use connect4::engine::{Engine as Connect4Engine};
use connect4::model_factory::{ModelFactory as Connect4ModelFactory};

fn main() -> Result<(), &'static str> {
    let run_name = "run-1";

    let model_factory = Connect4ModelFactory::new();
    let game_engine = Connect4Engine::new();
    
    self_evaluate::SelfEvaluate::evaluate(run_name)
}
