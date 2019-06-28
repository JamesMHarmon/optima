use std::io::Write;
use std::io::Read;
use std::fs::{create_dir_all, OpenOptions};
use std::path::{Path,PathBuf};
use serde::{Serialize, Deserialize};
use serde::de::DeserializeOwned;

use super::analytics::GameAnalytics;
use super::engine::GameEngine;
use super::game_state::GameState;
use super::self_play;
use super::self_play_persistance::{SelfPlayPersistance};
use super::model::{Model, ModelFactory};

// game/run/iteration/
//                  ./games
//                  ./nets
pub struct SelfLearn<'a, S, A, E, M, F>
where
    S: GameState,
    A: Clone + Eq + Serialize,
    E: 'a + GameEngine<State=S,Action=A>,
    M: 'a + Model + GameAnalytics<State=S,Action=A>,
    F: ModelFactory<M=M>
{
    options: SelfLearnOptions,
    game_name: String,
    run_name: String,
    run_directory: PathBuf,
    model_factory: F,
    latest_model: M,
    game_engine: &'a E
}

#[derive(Serialize, Deserialize, Debug)]
pub struct SelfLearnOptions {
    pub number_of_games_per_net: usize,
    pub moving_window_size: usize,
    pub train_ratio: f64,
    pub train_batch_size: usize,
    pub epochs: usize,
    pub learning_rate: f64,
    pub policy_loss_weight: f64,
    pub value_loss_weight: f64,
    pub temperature: f64,
    pub visits: usize,
    pub cpuct: f64,
    pub number_of_filters: usize,
    pub number_of_residual_blocks: usize
}

impl<'a, S, A, E, M, F> SelfLearn<'a, S, A, E, M, F>
where
    S: GameState,
    A: Clone + Eq + DeserializeOwned + Serialize,
    E: 'a + GameEngine<State=S,Action=A>,
    M: 'a + Model + GameAnalytics<State=S,Action=A>,
    F: ModelFactory<M=M>
{
    pub fn create(
        game_name: String,
        run_name: String,
        model_factory: &F
    ) -> Result<(), &'static str> {
        if game_name.contains("_") {
            return Err("game_name cannot contain any '_' characters");
        }

        if run_name.contains("_") {
            return Err("run_name cannot contain any '_' characters");
        }

        let default_options = SelfLearnOptions {
            number_of_games_per_net: 1000,
            moving_window_size: 10000,
            train_ratio: 0.9,
            train_batch_size: 512,
            epochs: 2,
            learning_rate: 0.001,
            policy_loss_weight: 1.0,
            value_loss_weight: 0.5,
            temperature: 1.0,
            visits: 800,
            cpuct: 4.0,
            number_of_filters: 128,
            number_of_residual_blocks: 5
        };

        SelfLearn::<S,A,E,M,F>::initialize_directories_and_files(&game_name, &run_name, &default_options)?;
        let model_name = Self::get_model_name(&game_name, &run_name, 1);

        model_factory.create(&model_name);

        Ok(())
    }

    pub fn from(
        game_name: String,
        run_name: String,
        model_factory: F,
        game_engine: &'a E
    ) -> Result<Self, &'static str> {
        let run_directory = Self::get_run_directory(&game_name, &run_name);
        let options = Self::get_config(&run_directory)?;
        let a_name = Self::get_model_name(&game_name, &run_name, 1);
        let latest_model = model_factory.get_latest(&a_name);

        Ok(Self {
            options,
            game_name,
            run_name,
            run_directory,
            model_factory,
            latest_model,
            game_engine
        })
    }

    pub fn learn(&mut self) -> Result<(), &'static str> {
        loop {
            let model_name = self.latest_model.get_name();
            let mut self_play_persistance = SelfPlayPersistance::new(
                &self.run_directory,
                model_name.to_owned()
            )?;

            let mut games = self_play_persistance.read::<A>()?;
            let number_of_games_per_net = self.options.number_of_games_per_net;

            while games.len() < number_of_games_per_net {
                let self_play_metrics = self_play::self_play(self.game_engine, &self.latest_model).unwrap();
                self_play_persistance.write(&self_play_metrics)?;
                games.push(self_play_metrics);
                println!("Played a game: {}", games.len());
            }

            // @TODO: Train a new model here.
            let new_model_name = Self::increment_model_name(&model_name);
            self.latest_model = self.model_factory.create(&new_model_name);
        }
    }

    fn get_model_name(game_name: &str, run_name: &str, model_number: usize) -> String {
        format!("{}_{}_{:0>5}", game_name, run_name, model_number)
    }

    fn increment_model_name(name: &str) -> String {
        let len = name.len();
        let head = &name[0..len-5];
        let tail = &name[len-5..];
        let count = tail.parse::<usize>().expect("Could not parse num from name");
        format!("{}{:0>5}", head, count + 1)
    }

    fn get_run_directory(game_name: &str, run_name: &str) -> PathBuf {
        PathBuf::from(format!("./{}_runs/{}", game_name, run_name))
    }

    fn get_config_path(run_directory: &Path) -> PathBuf {
        run_directory.join("config.json")
    }

    fn get_config(run_directory: &Path) -> Result<SelfLearnOptions, &'static str> {
        let config_path = Self::get_config_path(run_directory);
        let mut file = OpenOptions::new()
            .read(true)
            .open(config_path)
            .expect("Couldn't load config file.");

        let mut config_file_contents = String::new();
        file.read_to_string(&mut config_file_contents).expect("Failed to read config file");
        let options: SelfLearnOptions = serde_json::from_str(&config_file_contents).expect("Failed to parse config file");
        Ok(options)
    }

    fn initialize_directories_and_files(game_name: &str, run_name: &str, options: &SelfLearnOptions) -> Result<PathBuf, &'static str> {
        let run_directory = SelfLearn::<S,A,E,M,F>::get_run_directory(game_name, run_name);
        create_dir_all(&run_directory).expect("Run already exists or unable to create directories");

        let config_path = Self::get_config_path(&run_directory);

        if config_path.exists() {
            return Err("Run already exists");
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
}
