mod ponder;

use model::model::ModelFactory;
use quoridor::engine::{Engine as QuoridorEngine};
use quoridor::model::{ModelFactory as QuoridorModelFactory};
use model::model_info::ModelInfo;
use failure::Error;

use ponder::PonderOptions;

#[tokio::main]
async fn main() -> Result<(), Error> {
    let game_name = "Quoridor";
    let run_name = "run-1";

    let model_factory = QuoridorModelFactory::new();
    let latest_model_info = model_factory.get_latest(&ModelInfo::new(
        game_name.to_owned(),
        run_name.to_owned(),
        1
    ))?;

    let model = model_factory.get(&latest_model_info);

    let game_engine = QuoridorEngine::new();

    let options = PonderOptions {
        cpuct_base: 19_652.0,
        cpuct_init: 1.25,
        cpuct_root_scaling: 2.0,
        visits: 800
    };

    ponder::Ponder::ponder(
        &model,
        &game_engine,
        &options,
    ).await?;

    Ok(())
}
