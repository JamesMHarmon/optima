#![feature(async_await)]

#[macro_use]
extern crate clap;

use clap::App;
use connect4::engine::{Engine as Connect4Engine};
use connect4::model::{ModelFactory as Connect4ModelFactory};
use quoridor::engine::{Engine as QuoridorEngine};
use quoridor::model::{ModelFactory as QuoridorModelFactory};
use self_learn::self_learn::{Options,ModelOptions,SelfLearn,SelfLearnOptions};
use self_evaluate::self_evaluate::{SelfEvaluateOptions};

use failure::Error;

const C4_NAME: &str = "Connect4";
const QUORIDOR_NAME: &str = "Quoridor";

fn main() -> Result<(), Error> {
    let yaml = load_yaml!("cli.yml");
    let matches = App::from_yaml(yaml).get_matches();

    if let Some(matches) = matches.subcommand_matches("init") {
        let game_name = matches.value_of("game").unwrap();
        let run_name = matches.value_of("run").unwrap();
        let options = get_options_from_matches(matches)?;

        if game_name == C4_NAME {
            return create_connect4(run_name, &options);
        } else if game_name == QUORIDOR_NAME {
            return create_quoridor(run_name, &options);
        } else {
            panic!("Game name not recognized");
        }
    } else if let Some(matches) = matches.subcommand_matches("run") {
        let game_name = matches.value_of("game").unwrap();
        let run_name = matches.value_of("run").unwrap();

        if game_name == C4_NAME {
            return run_connect4(run_name);
        } else if game_name == QUORIDOR_NAME {
            return run_quoridor(run_name);
        } else {
            panic!("Game name not recognized");
        }
    }

    Ok(())
}

fn create_connect4(run_name: &str, options: &Options) -> Result<(), Error> {
    let model_factory = Connect4ModelFactory::new();

    SelfLearn::<_,_,Connect4Engine>::create(
        C4_NAME.to_owned(),
        run_name.to_owned(),
        &model_factory,
        options
    )
}

fn run_connect4(run_name: &str) -> Result<(), Error> {
    let model_factory = Connect4ModelFactory::new();
    let game_engine = Connect4Engine::new();

    let mut runner = SelfLearn::from(
        C4_NAME.to_owned(),
        run_name.to_owned(),
        &game_engine
    )?;

    runner.learn(&model_factory)?;

    Ok(())
}

fn create_quoridor(run_name: &str, options: &Options) -> Result<(), Error> {
    let model_factory = QuoridorModelFactory::new();

    SelfLearn::<_,_,QuoridorEngine>::create(
        QUORIDOR_NAME.to_owned(),
        run_name.to_owned(),
        &model_factory,
        options
    )
}

fn run_quoridor(run_name: &str) -> Result<(), Error> {
    let model_factory = QuoridorModelFactory::new();
    let game_engine = QuoridorEngine::new();

    let mut runner = SelfLearn::from(
        QUORIDOR_NAME.to_owned(),
        run_name.to_owned(),
        &game_engine
    )?;

    runner.learn(&model_factory)?;

    Ok(())
}

fn get_options_from_matches(matches: &clap::ArgMatches) -> Result<Options, Error> {
    let mut self_learn_options = SelfLearnOptions {
        number_of_games_per_net: 32_000,
        self_play_batch_size: 256,
        moving_window_size: 500_000,
        max_moving_window_percentage: 0.5,
        position_sample_percentage: 0.0512,
        train_ratio: 0.9,
        train_batch_size: 512,
        epochs: 1,
        learning_rate: 0.1,
        policy_loss_weight: 1.0,
        value_loss_weight: 0.5,
        temperature: 1.2,
        temperature_max_actions: 30,
        temperature_post_max_actions: 0.45,
        visits: 800,
        cpuct_base: 19_652.0,
        cpuct_init: 1.25,
        cpuct_root_scaling: 2.0,
        alpha: 0.3,
        epsilon: 0.25
    };

    if let Some(number_of_games_per_net) = matches.value_of("number_of_games_per_net") { self_learn_options.number_of_games_per_net = number_of_games_per_net.parse()? };
    if let Some(self_play_batch_size) = matches.value_of("self_play_batch_size") { self_learn_options.self_play_batch_size = self_play_batch_size.parse()? };
    if let Some(moving_window_size) = matches.value_of("moving_window_size") { self_learn_options.moving_window_size = moving_window_size.parse()? };
    if let Some(max_moving_window_percentage) = matches.value_of("max_moving_window_percentage") { self_learn_options.max_moving_window_percentage = max_moving_window_percentage.parse()? };
    if let Some(position_sample_percentage) = matches.value_of("position_sample_percentage") { self_learn_options.position_sample_percentage = position_sample_percentage.parse()? };
    if let Some(train_ratio) = matches.value_of("train_ratio") { self_learn_options.train_ratio = train_ratio.parse()? };
    if let Some(train_batch_size) = matches.value_of("train_batch_size") { self_learn_options.train_batch_size = train_batch_size.parse()? };
    if let Some(epochs) = matches.value_of("epochs") { self_learn_options.epochs = epochs.parse()? };
    if let Some(learning_rate) = matches.value_of("learning_rate") { self_learn_options.learning_rate = learning_rate.parse()? };
    if let Some(policy_loss_weight) = matches.value_of("policy_loss_weight") { self_learn_options.policy_loss_weight = policy_loss_weight.parse()? };
    if let Some(value_loss_weight) = matches.value_of("value_loss_weight") { self_learn_options.value_loss_weight = value_loss_weight.parse()? };
    if let Some(temperature) = matches.value_of("temperature") { self_learn_options.temperature = temperature.parse()? };
    if let Some(temperature_max_actions) = matches.value_of("temperature_max_actions") { self_learn_options.temperature_max_actions = temperature_max_actions.parse()? };
    if let Some(temperature_post_max_actions) = matches.value_of("temperature_post_max_actions") { self_learn_options.temperature_post_max_actions = temperature_post_max_actions.parse()? };
    if let Some(visits) = matches.value_of("visits") { self_learn_options.visits = visits.parse()? };
    if let Some(cpuct_base) = matches.value_of("cpuct_base") { self_learn_options.cpuct_base = cpuct_base.parse()? };
    if let Some(cpuct_init) = matches.value_of("cpuct_init") { self_learn_options.cpuct_init = cpuct_init.parse()? };
    if let Some(cpuct_root_scaling) = matches.value_of("cpuct_root_scaling") { self_learn_options.cpuct_root_scaling = cpuct_root_scaling.parse()? };
    if let Some(alpha) = matches.value_of("alpha") { self_learn_options.alpha = alpha.parse()? };
    if let Some(epsilon) = matches.value_of("epsilon") { self_learn_options.epsilon = epsilon.parse()? };

    let mut model_options = ModelOptions {
        number_of_filters: 64,
        number_of_residual_blocks: 5
    };

    if let Some(number_of_filters) = matches.value_of("number_of_filters") { model_options.number_of_filters = number_of_filters.parse()? };
    if let Some(number_of_residual_blocks) = matches.value_of("number_of_residual_blocks") { model_options.number_of_residual_blocks = number_of_residual_blocks.parse()? };

    let self_evaluate_options = SelfEvaluateOptions {
        num_games: 1000,
        temperature: self_learn_options.temperature,
        temperature_max_actions: self_learn_options.temperature_max_actions,
        temperature_post_max_actions: self_learn_options.temperature_post_max_actions,
        visits: self_learn_options.visits,
        cpuct_base: self_learn_options.cpuct_base,
        cpuct_init: self_learn_options.cpuct_init,
        cpuct_root_scaling: self_learn_options.cpuct_root_scaling
    };

    Ok(Options {
        self_learn: self_learn_options,
        self_evaluate: self_evaluate_options,
        model: model_options
    })
}
