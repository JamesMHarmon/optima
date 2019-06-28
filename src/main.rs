extern crate quoridor;

use quoridor::connect4::engine::{Engine as Connect4Engine};
use quoridor::connect4::model_factory::{ModelFactory as Connect4ModelFactory};
use quoridor::self_learn::{SelfLearn};

fn main() -> Result<(), &'static str>{
    let model_factory = Connect4ModelFactory::new();
    let game_engine = Connect4Engine::new();

    SelfLearn::<_,_,Connect4Engine,_,_>::create(
        "Connect4".to_string(), 
        "Run-2".to_string(),
        &model_factory
    ).ok();

    let runner = SelfLearn::from(
        "Connect4".to_string(),
        "Run-2".to_string(),
        model_factory,
        &game_engine
    );

    runner?.learn()?;

    Ok(())
}

