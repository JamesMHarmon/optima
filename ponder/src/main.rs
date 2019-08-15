#![feature(async_await)]

mod ponder;

use model::model::ModelFactory;
use connect4::engine::{Engine as Connect4Engine};
use connect4::model_factory::{ModelFactory as Connect4ModelFactory};
use model::model_info::ModelInfo;

use ponder::PonderOptions;

#[tokio::main]
async fn main() -> Result<(), &'static str> {
    let game_name = "Connect4";
    let run_name = "run-1";

    let model_factory = Connect4ModelFactory::new();
    let model = model_factory.get_latest(&ModelInfo::new(
        game_name.to_owned(),
        run_name.to_owned(),
        1
    ));
    let game_engine = Connect4Engine::new();

    let options = PonderOptions {
        cpuct_base: 19_652.0,
        cpuct_init: 1.25,
        visits: 800
    };

    ponder::Ponder::ponder(
        &model,
        &game_engine,
        &options,
    ).await?;

    Ok(())
}
