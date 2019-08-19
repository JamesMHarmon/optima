#![feature(async_await)]

mod self_evaluate;
mod self_evaluate_persistance;
mod constants;

use connect4::engine::{Engine as Connect4Engine};
use connect4::model_factory::{ModelFactory as Connect4ModelFactory};
use model::model_info::ModelInfo;

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

    for model_num in 1..100 {
        let model_1_info = ModelInfo::new(game_name.to_owned(), run_name.to_owned(), model_num);
        let model_2_info = ModelInfo::new(game_name.to_owned(), run_name.to_owned(), model_num + 1);

        self_evaluate::SelfEvaluate::evaluate(
            &model_1_info,
            &model_2_info,
            &model_factory,
            &game_engine,
            1000,
            &options,
        )?;
    }

    Ok(())
}