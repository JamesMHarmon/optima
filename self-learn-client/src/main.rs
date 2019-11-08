#[macro_use]
extern crate clap;

use std::path::Path;
use std::path::PathBuf;
use std::fs::{create_dir_all, OpenOptions};
use std::io::Write;
use std::io::Read;
use serde::{Serialize, Deserialize};
use model::model::{ModelOptions,ModelFactory};
use model::model_info::ModelInfo;
use clap::App;
use connect4::engine::{Engine as Connect4Engine};
use connect4::model::{ModelFactory as Connect4ModelFactory};
use quoridor::engine::{Engine as QuoridorEngine};
use quoridor::model::{ModelFactory as QuoridorModelFactory};
use arimaa::engine::{Engine as ArimaaEngine};
use arimaa::model::{ModelFactory as ArimaaModelFactory};
use self_learn::self_learn::{SelfLearn,SelfLearnOptions};
use self_evaluate::self_evaluate::{SelfEvaluateOptions};
use ponder::ponder::{Ponder,PonderOptions};

use failure::{Error,format_err};

const C4_NAME: &str = "Connect4";
const QUORIDOR_NAME: &str = "Quoridor";
const ARIMAA_NAME: &str = "Arimaa";

#[derive(Serialize, Deserialize, Debug)]
pub struct Options {
    pub self_learn: SelfLearnOptions,
    pub self_evaluate: SelfEvaluateOptions,
    pub model: ModelOptions,
    pub ponder: PonderOptions
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    let yaml = load_yaml!("cli.yml");
    let matches = App::from_yaml(yaml).get_matches();

    if let Some(matches) = matches.subcommand_matches("init") {
        let game_name = matches.value_of("game").unwrap();
        let run_name = matches.value_of("run").unwrap();
        let mut default_options = get_default_options()?;
        update_model_options_from_matches(&mut default_options.model, matches)?;
        update_self_learn_options_from_matches(&mut default_options.self_learn, matches)?;
        update_self_evaluate_options_from_matches(&mut default_options.self_evaluate, matches)?;
        update_ponder_options_from_matches(&mut default_options.ponder, matches)?;
        let options = default_options;

        create(game_name, run_name, &options)?;

        if game_name == C4_NAME {
            return create_connect4(run_name, &options.model);
        } else if game_name == QUORIDOR_NAME {
            return create_quoridor(run_name, &options.model);
        } else if game_name == ARIMAA_NAME {
            return create_arimaa(run_name, &options.model);
        } else {
            panic!("Game name not recognized");
        }
    } else if let Some(matches) = matches.subcommand_matches("run") {
        let game_name = matches.value_of("game").unwrap();
        let run_name = matches.value_of("run").unwrap();
        let options = get_options(game_name, run_name)?;

        if game_name == C4_NAME {
            return run_connect4(run_name, &options.self_learn, &options.self_evaluate);
        } else if game_name == QUORIDOR_NAME {
            return run_quoridor(run_name, &options.self_learn, &options.self_evaluate);
        } else if game_name == ARIMAA_NAME {
            return run_arimaa(run_name, &options.self_learn, &options.self_evaluate);
        } else {
            panic!("Game name not recognized");
        }
    } else if let Some(matches) = matches.subcommand_matches("evaluate") {
        let game_name = matches.value_of("game").unwrap();
        let run_name = matches.value_of("run").unwrap();
        let mut options = get_options(game_name, run_name)?.self_evaluate;
        let (model_1_num, model_2_num) = update_self_evaluate_options_from_matches(&mut options, matches)?;
        let model_1_num: Option<usize> = model_1_num.map(|v| v.parse().expect("Model number not a valid int"));
        let model_2_num: Option<usize> = model_2_num.map(|v| v.parse().expect("Model number not a valid int"));

        if game_name == C4_NAME {
            return evaluate_connect4(run_name, model_1_num, model_2_num, &options);
        } else if game_name == QUORIDOR_NAME {
            return evaluate_quoridor(run_name, model_1_num, model_2_num, &options);
        } else if game_name == ARIMAA_NAME {
            return evaluate_arimaa(run_name, model_1_num, model_2_num, &options);
        } else {
            panic!("Game name not recognized");
        }
    } else if let Some(matches) = matches.subcommand_matches("ponder") {
        let game_name = matches.value_of("game").unwrap();
        let run_name = matches.value_of("run").unwrap();
        let mut options = get_options(game_name, run_name)?.ponder;
        let model_num = update_ponder_options_from_matches(&mut options, matches)?;
        let model_num: Option<usize> = model_num.map(|v| v.parse().expect("Model number not a valid int"));

        if game_name == C4_NAME {
            return ponder_connect4(run_name, model_num, &options).await;
        } else if game_name == QUORIDOR_NAME {
            return ponder_quoridor(run_name, model_num, &options).await;
        } else if game_name == ARIMAA_NAME {
            return ponder_arimaa(run_name, model_num, &options).await;
        } else {
            panic!("Game name not recognized");
        }
    }

    Ok(())
}

fn create_connect4(run_name: &str, options: &ModelOptions) -> Result<(), Error> {
    let model_factory = Connect4ModelFactory::new();

    SelfLearn::<_,_,_,Connect4Engine>::create(
        C4_NAME.to_owned(),
        run_name.to_owned(),
        &model_factory,
        options
    )
}

fn run_connect4(run_name: &str, self_learn_options: &SelfLearnOptions, self_evaluate_options: &SelfEvaluateOptions) -> Result<(), Error> {
    let model_factory = Connect4ModelFactory::new();
    let game_engine = Connect4Engine::new();
    let run_directory = get_run_directory(C4_NAME, run_name);

    let mut runner = SelfLearn::from(
        C4_NAME.to_owned(),
        run_name.to_owned(),
        &game_engine,
        &run_directory,
        self_learn_options,
        self_evaluate_options
    )?;

    runner.learn(&model_factory)?;

    Ok(())
}

fn evaluate_connect4(run_name: &str, model_1_num: Option<usize>, model_2_num: Option<usize>, options: &SelfEvaluateOptions) -> Result<(), Error> {
    let model_factory = Connect4ModelFactory::new();
    let game_engine = Connect4Engine::new();
    let model_info = ModelInfo::new(C4_NAME.to_owned(), run_name.to_owned(), 1);
    let latest_model_num = model_factory.get_latest(&model_info)?;

    let model_1_num = model_1_num.unwrap_or_else(|| latest_model_num.get_model_num() - 1);
    let model_2_num = model_2_num.unwrap_or_else(|| latest_model_num.get_model_num());

    let model_1_info = ModelInfo::new(C4_NAME.to_owned(), run_name.to_owned(), model_1_num);
    let model_2_info = ModelInfo::new(C4_NAME.to_owned(), run_name.to_owned(), model_2_num);

    let model_1 = model_factory.get(&model_1_info);
    let model_2 = model_factory.get(&model_2_info);

    self_evaluate::self_evaluate::SelfEvaluate::evaluate(
        &vec!(model_1, model_2),
        &game_engine,
        options
    )?;

    Ok(())
}

async fn ponder_connect4(run_name: &str, model_num: Option<usize>, options: &PonderOptions) -> Result<(), Error> {
    let model_factory = Connect4ModelFactory::new();
    let game_engine = Connect4Engine::new();
    let model_info = ModelInfo::new(C4_NAME.to_owned(), run_name.to_owned(), 1);
    let latest_model_info = model_factory.get_latest(&model_info)?;

    let model_num = model_num.unwrap_or_else(|| latest_model_info.get_model_num());

    let model_info = ModelInfo::new(C4_NAME.to_owned(), run_name.to_owned(), model_num);

    let model = model_factory.get(&model_info);

    Ponder::ponder(
        &model,
        &game_engine,
        options
    ).await?;

    Ok(())
}

fn create(game_name: &str, run_name: &str, options: &Options) -> Result<(), Error> {
    if game_name.contains("_") {
        return Err(format_err!("game_name cannot contain any '_' characters"));
    }

    if run_name.contains("_") {
        return Err(format_err!("run_name cannot contain any '_' characters"));
    }

    initialize_directories_and_files(&game_name, &run_name, &options)?;

    Ok(())
}

fn create_quoridor(run_name: &str, options: &ModelOptions) -> Result<(), Error> {
    let model_factory = QuoridorModelFactory::new();

    SelfLearn::<_,_,_,QuoridorEngine>::create(
        QUORIDOR_NAME.to_owned(),
        run_name.to_owned(),
        &model_factory,
        options
    )
}

fn run_quoridor(run_name: &str, self_learn_options: &SelfLearnOptions, self_evaluate_options: &SelfEvaluateOptions) -> Result<(), Error> {
    let model_factory = QuoridorModelFactory::new();
    let game_engine = QuoridorEngine::new();
    let run_directory = get_run_directory(QUORIDOR_NAME, run_name);

    let mut runner = SelfLearn::from(
        QUORIDOR_NAME.to_owned(),
        run_name.to_owned(),
        &game_engine,
        &run_directory,
        self_learn_options,
        self_evaluate_options
    )?;

    runner.learn(&model_factory)?;

    Ok(())
}

fn evaluate_quoridor(run_name: &str, model_1_num: Option<usize>, model_2_num: Option<usize>, options: &SelfEvaluateOptions) -> Result<(), Error> {
    let model_factory = QuoridorModelFactory::new();
    let game_engine = QuoridorEngine::new();
    let model_info = ModelInfo::new(QUORIDOR_NAME.to_owned(), run_name.to_owned(), 1);
    let latest_model_num = model_factory.get_latest(&model_info)?;

    let model_1_num = model_1_num.unwrap_or_else(|| latest_model_num.get_model_num() - 1);
    let model_2_num = model_2_num.unwrap_or_else(|| latest_model_num.get_model_num());

    let model_1_info = ModelInfo::new(QUORIDOR_NAME.to_owned(), run_name.to_owned(), model_1_num);
    let model_2_info = ModelInfo::new(QUORIDOR_NAME.to_owned(), run_name.to_owned(), model_2_num);

    let model_1 = model_factory.get(&model_1_info);
    let model_2 = model_factory.get(&model_2_info);

    self_evaluate::self_evaluate::SelfEvaluate::evaluate(
        &vec!(model_1, model_2),
        &game_engine,
        options
    )?;

    Ok(())
}

async fn ponder_quoridor(run_name: &str, model_num: Option<usize>, options: &PonderOptions) -> Result<(), Error> {
    let model_factory = QuoridorModelFactory::new();
    let game_engine = QuoridorEngine::new();
    let model_info = ModelInfo::new(QUORIDOR_NAME.to_owned(), run_name.to_owned(), 1);
    let latest_model_info = model_factory.get_latest(&model_info)?;

    let model_num = model_num.unwrap_or_else(|| latest_model_info.get_model_num());

    let model_info = ModelInfo::new(QUORIDOR_NAME.to_owned(), run_name.to_owned(), model_num);

    let model = model_factory.get(&model_info);

    Ponder::ponder(
        &model,
        &game_engine,
        options
    ).await?;

    Ok(())
}

fn create_arimaa(run_name: &str, options: &ModelOptions) -> Result<(), Error> {
    let model_factory = ArimaaModelFactory::new();

    SelfLearn::<_,_,_,ArimaaEngine>::create(
        ARIMAA_NAME.to_owned(),
        run_name.to_owned(),
        &model_factory,
        options
    )
}

fn run_arimaa(run_name: &str, self_learn_options: &SelfLearnOptions, self_evaluate_options: &SelfEvaluateOptions) -> Result<(), Error> {
    let model_factory = ArimaaModelFactory::new();
    let game_engine = ArimaaEngine::new();
    let run_directory = get_run_directory(ARIMAA_NAME, run_name);

    let mut runner = SelfLearn::from(
        ARIMAA_NAME.to_owned(),
        run_name.to_owned(),
        &game_engine,
        &run_directory,
        self_learn_options,
        self_evaluate_options
    )?;

    runner.learn(&model_factory)?;

    Ok(())
}

fn evaluate_arimaa(run_name: &str, model_1_num: Option<usize>, model_2_num: Option<usize>, options: &SelfEvaluateOptions) -> Result<(), Error> {
    let model_factory = ArimaaModelFactory::new();
    let game_engine = ArimaaEngine::new();
    let model_info = ModelInfo::new(ARIMAA_NAME.to_owned(), run_name.to_owned(), 1);
    let latest_model_num = model_factory.get_latest(&model_info)?;

    let model_1_num = model_1_num.unwrap_or_else(|| latest_model_num.get_model_num() - 1);
    let model_2_num = model_2_num.unwrap_or_else(|| latest_model_num.get_model_num());

    let model_1_info = ModelInfo::new(ARIMAA_NAME.to_owned(), run_name.to_owned(), model_1_num);
    let model_2_info = ModelInfo::new(ARIMAA_NAME.to_owned(), run_name.to_owned(), model_2_num);

    let model_1 = model_factory.get(&model_1_info);
    let model_2 = model_factory.get(&model_2_info);

    self_evaluate::self_evaluate::SelfEvaluate::evaluate(
        &vec!(model_1, model_2),
        &game_engine,
        options
    )?;

    Ok(())
}

async fn ponder_arimaa(run_name: &str, model_num: Option<usize>, options: &PonderOptions) -> Result<(), Error> {
    let model_factory = ArimaaModelFactory::new();
    let game_engine = ArimaaEngine::new();
    let model_info = ModelInfo::new(ARIMAA_NAME.to_owned(), run_name.to_owned(), 1);
    let latest_model_info = model_factory.get_latest(&model_info)?;

    let model_num = model_num.unwrap_or_else(|| latest_model_info.get_model_num());

    let model_info = ModelInfo::new(ARIMAA_NAME.to_owned(), run_name.to_owned(), model_num);

    let model = model_factory.get(&model_info);

    Ponder::ponder(
        &model,
        &game_engine,
        options
    ).await?;

    Ok(())
}

fn get_default_options() -> Result<Options, Error> {
    let self_learn_options = SelfLearnOptions {
        number_of_games_per_net: 32_000,
        self_play_batch_size: 256,
        moving_window_size: 500_000,
        max_moving_window_percentage: 0.5,
        position_sample_percentage: 0.0512,
        exclude_drawn_games: false,
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
        parallelism: 32,
        fpu: 0.0,
        fpu_root: 1.0,
        cpuct_base: 19_652.0,
        cpuct_init: 1.25,
        cpuct_root_scaling: 2.0,
        alpha: 0.3,
        epsilon: 0.25
    };

    let model_options = ModelOptions {
        number_of_filters: 64,
        number_of_residual_blocks: 5
    };

    let self_evaluate_options = SelfEvaluateOptions {
        num_games: 200,
        batch_size: self_learn_options.self_play_batch_size,
        parallelism: self_learn_options.parallelism * 4,
        temperature: self_learn_options.temperature,
        temperature_max_actions: self_learn_options.temperature_max_actions,
        temperature_post_max_actions: self_learn_options.temperature_post_max_actions,
        visits: self_learn_options.visits * 5,
        fpu: self_learn_options.fpu,
        fpu_root: self_learn_options.fpu_root,
        cpuct_base: self_learn_options.cpuct_base,
        cpuct_init: self_learn_options.cpuct_init,
        cpuct_root_scaling: self_learn_options.cpuct_root_scaling
    };

    let ponder_options = PonderOptions {
        visits: self_learn_options.visits * 50,
        parallelism: self_learn_options.parallelism * 4,
        fpu: self_learn_options.fpu,
        fpu_root: self_learn_options.fpu_root,
        cpuct_base: self_learn_options.cpuct_base,
        cpuct_init: self_learn_options.cpuct_init,
        cpuct_root_scaling: self_learn_options.cpuct_root_scaling
    };

    Ok(Options {
        self_learn: self_learn_options,
        self_evaluate: self_evaluate_options,
        model: model_options,
        ponder: ponder_options
    })
}

fn update_self_learn_options_from_matches(options: &mut SelfLearnOptions, matches: &clap::ArgMatches) -> Result<(), Error> {
    if let Some(number_of_games_per_net) = matches.value_of("number_of_games_per_net") { options.number_of_games_per_net = number_of_games_per_net.parse()? };
    if let Some(self_play_batch_size) = matches.value_of("self_play_batch_size") { options.self_play_batch_size = self_play_batch_size.parse()? };
    if let Some(moving_window_size) = matches.value_of("moving_window_size") { options.moving_window_size = moving_window_size.parse()? };
    if let Some(max_moving_window_percentage) = matches.value_of("max_moving_window_percentage") { options.max_moving_window_percentage = max_moving_window_percentage.parse()? };
    if let Some(position_sample_percentage) = matches.value_of("position_sample_percentage") { options.position_sample_percentage = position_sample_percentage.parse()? };
    if let Some(exclude_drawn_games) = matches.value_of("exclude_drawn_games") { options.exclude_drawn_games = exclude_drawn_games.parse()? };
    if let Some(train_ratio) = matches.value_of("train_ratio") { options.train_ratio = train_ratio.parse()? };
    if let Some(train_batch_size) = matches.value_of("train_batch_size") { options.train_batch_size = train_batch_size.parse()? };
    if let Some(epochs) = matches.value_of("epochs") { options.epochs = epochs.parse()? };
    if let Some(learning_rate) = matches.value_of("learning_rate") { options.learning_rate = learning_rate.parse()? };
    if let Some(policy_loss_weight) = matches.value_of("policy_loss_weight") { options.policy_loss_weight = policy_loss_weight.parse()? };
    if let Some(value_loss_weight) = matches.value_of("value_loss_weight") { options.value_loss_weight = value_loss_weight.parse()? };
    if let Some(temperature) = matches.value_of("temperature") { options.temperature = temperature.parse()? };
    if let Some(temperature_max_actions) = matches.value_of("temperature_max_actions") { options.temperature_max_actions = temperature_max_actions.parse()? };
    if let Some(temperature_post_max_actions) = matches.value_of("temperature_post_max_actions") { options.temperature_post_max_actions = temperature_post_max_actions.parse()? };
    if let Some(visits) = matches.value_of("visits") { options.visits = visits.parse()? };
    if let Some(fpu) = matches.value_of("fpu") { options.fpu = fpu.parse()? };
    if let Some(fpu_root) = matches.value_of("fpu_root") { options.fpu_root = fpu_root.parse()? };
    if let Some(cpuct_base) = matches.value_of("cpuct_base") { options.cpuct_base = cpuct_base.parse()? };
    if let Some(cpuct_init) = matches.value_of("cpuct_init") { options.cpuct_init = cpuct_init.parse()? };
    if let Some(cpuct_root_scaling) = matches.value_of("cpuct_root_scaling") { options.cpuct_root_scaling = cpuct_root_scaling.parse()? };
    if let Some(alpha) = matches.value_of("alpha") { options.alpha = alpha.parse()? };
    if let Some(epsilon) = matches.value_of("epsilon") { options.epsilon = epsilon.parse()? };

    Ok(())
}

fn update_model_options_from_matches(options: &mut ModelOptions, matches: &clap::ArgMatches) -> Result<(), Error> {
    if let Some(number_of_filters) = matches.value_of("number_of_filters") { options.number_of_filters = number_of_filters.parse()? };
    if let Some(number_of_residual_blocks) = matches.value_of("number_of_residual_blocks") { options.number_of_residual_blocks = number_of_residual_blocks.parse()? };

    Ok(())
}

fn update_self_evaluate_options_from_matches(options: &mut SelfEvaluateOptions, matches: &clap::ArgMatches) -> Result<(Option<String>, Option<String>), Error> {
    if let Some(temperature) = matches.value_of("temperature") { options.temperature = temperature.parse()? };
    if let Some(temperature_max_actions) = matches.value_of("temperature_max_actions") { options.temperature_max_actions = temperature_max_actions.parse()? };
    if let Some(temperature_post_max_actions) = matches.value_of("temperature_post_max_actions") { options.temperature_post_max_actions = temperature_post_max_actions.parse()? };
    if let Some(visits) = matches.value_of("visits") { options.visits = visits.parse()? };
    if let Some(fpu) = matches.value_of("fpu") { options.fpu = fpu.parse()? };
    if let Some(fpu_root) = matches.value_of("fpu_root") { options.fpu_root = fpu_root.parse()? };
    if let Some(cpuct_base) = matches.value_of("cpuct_base") { options.cpuct_base = cpuct_base.parse()? };
    if let Some(cpuct_init) = matches.value_of("cpuct_init") { options.cpuct_init = cpuct_init.parse()? };
    if let Some(cpuct_root_scaling) = matches.value_of("cpuct_root_scaling") { options.cpuct_root_scaling = cpuct_root_scaling.parse()? };
    if let Some(num_games) = matches.value_of("num_games") { options.num_games = num_games.parse()? };

    Ok((matches.value_of("model_1").map(|s| s.to_owned()), matches.value_of("model_2").map(|s| s.to_owned())))
}

fn update_ponder_options_from_matches(options: &mut PonderOptions, matches: &clap::ArgMatches) -> Result<Option<String>, Error> {
    if let Some(visits) = matches.value_of("visits") { options.visits = visits.parse()? };
    if let Some(fpu) = matches.value_of("fpu") { options.fpu = fpu.parse()? };
    if let Some(fpu_root) = matches.value_of("fpu_root") { options.fpu_root = fpu_root.parse()? };
    if let Some(cpuct_base) = matches.value_of("cpuct_base") { options.cpuct_base = cpuct_base.parse()? };
    if let Some(cpuct_init) = matches.value_of("cpuct_init") { options.cpuct_init = cpuct_init.parse()? };
    if let Some(cpuct_root_scaling) = matches.value_of("cpuct_root_scaling") { options.cpuct_root_scaling = cpuct_root_scaling.parse()? };

    Ok(matches.value_of("model").map(|s| s.to_owned()))
}

fn get_options(game_name: &str, run_name: &str) -> Result<Options, Error> {
    let run_directory = get_run_directory(&game_name, &run_name);
    get_config(&run_directory)
}

fn get_run_directory(game_name: &str, run_name: &str) -> PathBuf {
    PathBuf::from(format!("./{}_runs/{}", game_name, run_name))
}

fn get_config_path(run_directory: &Path) -> PathBuf {
    run_directory.join("config.json")
}

fn get_config(run_directory: &Path) -> Result<Options, Error> {
    let config_path = get_config_path(run_directory);
    let mut file = OpenOptions::new()
        .read(true)
        .open(config_path)
        .expect("Couldn't load config file.");

    let mut config_file_contents = String::new();
    file.read_to_string(&mut config_file_contents).expect("Failed to read config file");
    let options: Options = serde_json::from_str(&config_file_contents).expect("Failed to parse config file");
    Ok(options)
}

fn initialize_directories_and_files(game_name: &str, run_name: &str, options: &Options) -> Result<PathBuf, Error> {
    let run_directory = get_run_directory(game_name, run_name);
    create_dir_all(&run_directory).expect("Run already exists or unable to create directories");

    let config_path = get_config_path(&run_directory);

    if config_path.exists() {
        return Err(format_err!("Run already exists"));
    }

    println!("{:?}", run_directory);
    println!("{:?}", config_path);

    let mut file = OpenOptions::new()
        .write(true)
        .create(true)
        .open(config_path)
        .expect("Couldn't open or create the config file");

    let serialized_options = serde_json::to_string_pretty(options).expect("Unable to serialize options");
    file.write(serialized_options.as_bytes()).expect("Unable to write options to file");

    Ok(run_directory)
}
