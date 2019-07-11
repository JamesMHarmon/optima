#![feature(async_await)]

#[macro_use]
extern crate clap;
extern crate quoridor;

use clap::App;
use tokio::runtime::current_thread;

use quoridor::connect4::engine::{Engine as Connect4Engine};
use quoridor::connect4::model_factory::{ModelFactory as Connect4ModelFactory};
use quoridor::self_learn::{SelfLearn,SelfLearnOptions};

const C4_NAME: &str = "Connect4";

fn main() -> Result<(), &'static str> {
    let yaml = load_yaml!("cli.yml");
    let matches = App::from_yaml(yaml).get_matches();

    if let Some(matches) = matches.subcommand_matches("init") {
        // let game_name = matches.value_of("game").unwrap();
        let run_name = matches.value_of("run").unwrap();
        let options = get_options_from_matches(matches)?;

        return create_connect4(run_name, &options);
    } else if let Some(matches) = matches.subcommand_matches("run") {
        // let game_name = matches.value_of("game").unwrap();
        let run_name = matches.value_of("run").unwrap();

        let result = current_thread::block_on_all(run_connect4(run_name));

        return result;
    }

    Ok(())
}

fn create_connect4(run_name: &str, options: &SelfLearnOptions) -> Result<(), &'static str> {
    let model_factory = Connect4ModelFactory::new();

    SelfLearn::<_,_,Connect4Engine,_>::create(
        C4_NAME.to_owned(),
        run_name.to_owned(),
        &model_factory,
        options
    )
}

async fn run_connect4(run_name: &str) -> Result<(), &'static str> {
    let model_factory = Connect4ModelFactory::new();
    let game_engine = Connect4Engine::new();

    let mut runner = SelfLearn::from(
        C4_NAME.to_owned(),
        run_name.to_owned(),
        model_factory,
        &game_engine
    )?;

    runner.learn().await
}

fn get_options_from_matches(matches: &clap::ArgMatches) -> Result<SelfLearnOptions, &'static str> {
    let mut options = SelfLearnOptions {
        number_of_games_per_net: 32_000,
        self_play_batch_size: 256,
        moving_window_size: 500_000,
        train_ratio: 0.9,
        train_batch_size: 512,
        epochs: 2,
        learning_rate: 0.001,
        policy_loss_weight: 1.0,
        value_loss_weight: 0.5,
        temperature: 1.2,
        temperature_max_actions: 30,
        temperature_post_max_actions: 0.45,
        visits: 800,
        cpuct_base: 19_652.0,
        cpuct_init: 1.25,
        alpha: 0.3,
        epsilon: 0.25,
        number_of_filters: 64,
        number_of_residual_blocks: 5
    };

    if let Some(number_of_games_per_net) = matches.value_of("number_of_games_per_net") { options.number_of_games_per_net = number_of_games_per_net.parse().map_err(|_| "Could not parse number_of_games_per_net")? };
    if let Some(self_play_batch_size) = matches.value_of("self_play_batch_size") { options.self_play_batch_size = self_play_batch_size.parse().map_err(|_| "Could not parse self_play_batch_size")? };
    if let Some(moving_window_size) = matches.value_of("moving_window_size") { options.moving_window_size = moving_window_size.parse().map_err(|_| "Could not parse moving_window_size")? };
    if let Some(train_ratio) = matches.value_of("train_ratio") { options.train_ratio = train_ratio.parse().map_err(|_| "Could not parse train_ratio")? };
    if let Some(train_batch_size) = matches.value_of("train_batch_size") { options.train_batch_size = train_batch_size.parse().map_err(|_| "Could not parse train_batch_size")? };
    if let Some(epochs) = matches.value_of("epochs") { options.epochs = epochs.parse().map_err(|_| "Could not parse epochs")? };
    if let Some(learning_rate) = matches.value_of("learning_rate") { options.learning_rate = learning_rate.parse().map_err(|_| "Could not parse learning_rate")? };
    if let Some(policy_loss_weight) = matches.value_of("policy_loss_weight") { options.policy_loss_weight = policy_loss_weight.parse().map_err(|_| "Could not parse policy_loss_weight")? };
    if let Some(value_loss_weight) = matches.value_of("value_loss_weight") { options.value_loss_weight = value_loss_weight.parse().map_err(|_| "Could not parse value_loss_weight")? };
    if let Some(temperature) = matches.value_of("temperature") { options.temperature = temperature.parse().map_err(|_| "Could not parse temperature")? };
    if let Some(temperature_max_actions) = matches.value_of("temperature_max_actions") { options.temperature_max_actions = temperature_max_actions.parse().map_err(|_| "Could not parse temperature_max_actions")? };
    if let Some(temperature_post_max_actions) = matches.value_of("temperature_post_max_actions") { options.temperature_post_max_actions = temperature_post_max_actions.parse().map_err(|_| "Could not parse temperature_post_max_actions")? };
    if let Some(visits) = matches.value_of("visits") { options.visits = visits.parse().map_err(|_| "Could not parse visits")? };
    if let Some(cpuct_base) = matches.value_of("cpuct_base") { options.cpuct_base = cpuct_base.parse().map_err(|_| "Could not parse cpuct_base")? };
    if let Some(cpuct_init) = matches.value_of("cpuct_init") { options.cpuct_init = cpuct_init.parse().map_err(|_| "Could not parse cpuct_init")? };
    if let Some(alpha) = matches.value_of("alpha") { options.alpha = alpha.parse().map_err(|_| "Could not parse alpha")? };
    if let Some(epsilon) = matches.value_of("epsilon") { options.epsilon = epsilon.parse().map_err(|_| "Could not parse epsilon")? };
    if let Some(number_of_filters) = matches.value_of("number_of_filters") { options.number_of_filters = number_of_filters.parse().map_err(|_| "Could not parse number_of_filters")? };
    if let Some(number_of_residual_blocks) = matches.value_of("number_of_residual_blocks") { options.number_of_residual_blocks = number_of_residual_blocks.parse().map_err(|_| "Could not parse number_of_residual_blocks")? };

    Ok(options)
}
