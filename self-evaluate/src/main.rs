#![feature(async_await)]

mod self_evaluate;

use connect4::engine::{Engine as Connect4Engine};
use connect4::model_factory::{ModelFactory as Connect4ModelFactory};

use self_evaluate::SelfEvaluateOptions;

fn main() -> Result<(), &'static str> {
    let game_name = "Connect4";
    let run_name = "run-1";

    let model_factory = Connect4ModelFactory::new();
    let game_engine = Connect4Engine::new();

    let options = SelfEvaluateOptions {
        cpuct_base: 19_652.0,
        cpuct_init: 1.25,
        temperature_max_actions: 16,
        temperature: 0.45,
        temperature_post_max_actions: 0.0,
        visits: 800
    };

    self_evaluate::SelfEvaluate::evaluate(
        game_name,
        run_name,
        model_factory,
        &game_engine,
        1000,
        3,
        &options,
    )
}
