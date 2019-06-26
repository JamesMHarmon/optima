extern crate quoridor;

use quoridor::connect4::engine::{Engine as Connect4Engine};
use quoridor::connect4::model::{Model as Connect4Model};
use quoridor::analysis_cache::{AnalysisCache};
use quoridor::self_learn::{SelfLearn,SelfLearnOptions};

fn main() -> Result<(), &'static str>{
    let model = Connect4Model::new();

    // SelfLearn::new(
    //     "Connect4".to_string(), 
    //     "Run_1".to_string(),
    //     model,
    //     SelfLearnOptions {
    //         number_of_games_per_net: 1000,
    //         moving_window_size: 10000,
    //         train_ratio: 0.9,
    //         train_batch_size: 512,
    //         epochs: 2,
    //         learning_rate: 0.001,
    //         policy_loss_weight: 1.0,
    //         value_loss_weight: 0.5,
    //         temperature: 1.0,
    //         visits: 800,
    //         cpuct: 4.0,
    //         number_of_filters: 128,
    //         number_of_residual_blocks: 5
    //     }
    // )?;

   SelfLearn::from("Connect4".to_string(), "Run-1".to_string(), model)?;

    Ok(())
}

