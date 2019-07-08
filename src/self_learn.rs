use std::fmt::Debug;
use rand::Rng;
use std::io::Write;
use std::io::Read;
use std::fs::{create_dir_all, OpenOptions};
use std::path::{Path,PathBuf};
use serde::{Serialize, Deserialize};
use serde::de::DeserializeOwned;

use super::analytics::GameAnalytics;
use super::engine::GameEngine;
use super::game_state::GameState;
use super::self_play::{self,SelfPlayOptions,SelfPlaySample};
use super::self_play_persistance::{SelfPlayPersistance};
use super::model::{Model, ModelFactory,TrainOptions};
use super::futures::join_all::join_all;

// game/run/iteration/
//                  ./games
//                  ./nets
pub struct SelfLearn<'a, S, A, E, M>
where
    S: GameState,
    A: Clone + Eq + Serialize + Unpin,
    E: 'a + GameEngine<State=S,Action=A>,
    M: 'a + Model + GameAnalytics<State=S,Action=A>
{
    options: SelfLearnOptions,
    run_directory: PathBuf,
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
    pub alpha: f64,
    pub epsilon: f64,
    pub number_of_filters: usize,
    pub number_of_residual_blocks: usize
}

impl<'a, S, A, E, M> SelfLearn<'a, S, A, E, M>
where
    S: GameState,
    A: Clone + Eq + DeserializeOwned + Serialize + Debug + Unpin,
    E: 'a + GameEngine<State=S,Action=A>,
    M: 'a + Model<State=S,Action=A> + GameAnalytics<State=S,Action=A>
{
    pub fn create<F>(
        game_name: String,
        run_name: String,
        model_factory: &F,
        options: &SelfLearnOptions
    ) -> Result<(), &'static str>
    where
        F: ModelFactory<M=M>
    {
        if game_name.contains("_") {
            return Err("game_name cannot contain any '_' characters");
        }

        if run_name.contains("_") {
            return Err("run_name cannot contain any '_' characters");
        }

        SelfLearn::<S,A,E,M>::initialize_directories_and_files(&game_name, &run_name, &options)?;
        let model_name = Self::get_model_name(&game_name, &run_name, 1);

        model_factory.create(&model_name, options.number_of_filters, options.number_of_residual_blocks);

        Ok(())
    }

    pub fn from<F>(
        game_name: String,
        run_name: String,
        model_factory: F,
        game_engine: &'a E
    ) -> Result<Self, &'static str> 
    where
        F: ModelFactory<M=M>
    {
        let run_directory = Self::get_run_directory(&game_name, &run_name);
        let options = Self::get_config(&run_directory)?;
        let a_name = Self::get_model_name(&game_name, &run_name, 1);
        let latest_model = model_factory.get_latest(&a_name);

        Ok(Self {
            options,
            run_directory,
            latest_model,
            game_engine
        })
    }

    pub async fn learn(&mut self) -> Result<(), &'static str> {
        let options = &self.options;
        let number_of_games_per_net = options.number_of_games_per_net;
        let self_play_options = SelfPlayOptions {
            epsilon: options.epsilon,
            alpha: options.alpha,
            cpuct: options.cpuct,
            temperature: options.temperature,
            visits: options.visits
        };

        loop {
            let latest_model = &self.latest_model;
            let model_name = latest_model.get_name();
            let mut self_play_persistance = SelfPlayPersistance::new(
                &self.run_directory,
                model_name.to_owned()
            )?;

            let mut num_games = self_play_persistance.read::<A>()?.len();

            while num_games < number_of_games_per_net {
                let game_batch_size = 512;
                let futures: Vec<_> = (0..game_batch_size).map(|_| Box::pin(async {
                    self_play::self_play(self.game_engine, latest_model, &self_play_options).await.unwrap()
                })).collect();

                let self_play_metrics = join_all(futures).await;

                for self_play_metric in self_play_metrics {
                    self_play_persistance.write(&self_play_metric).unwrap();
                }

                num_games += game_batch_size;
                println!("Played a game: {}", num_games);
            }

            self.latest_model = Self::train_model(
                latest_model,
                &self_play_persistance,
                &self.game_engine,
                options
            )?;
        }
    }

    fn train_model(model: &M, self_play_persistance: &SelfPlayPersistance, game_engine: &E, options: &SelfLearnOptions) -> Result<M, &'static str> {
        let source_model_name = &model.get_name();
        let new_model_name = Self::increment_model_name(source_model_name);
        let metric_iter = self_play_persistance.read_all_reverse_iter::<A>()?;
        let mut rng = rand::thread_rng();

        let sample_metrics: Vec<SelfPlaySample<S, A>> = metric_iter
            .take(options.moving_window_size)
            .map(|m| {
                let score = m.score();
                let mut analysis = m.take_analysis();
                let l = analysis.len();
                let i = rng.gen_range(0, l);
                let sample_is_p1 = i % 2 == 0;
                let score = score * if sample_is_p1 { 1.0 } else { -1.0 };
                let game_state = analysis.iter().take(i + 1).fold(S::initial(), |s,m| game_engine.take_action(&s, &m.0));

                SelfPlaySample {
                    game_state,
                    score,
                    policy: analysis.remove(i).1
                }
            })
            .collect();

        Ok(model.train(
            &new_model_name,
            &sample_metrics,
            &TrainOptions {
                train_ratio: options.train_ratio,
                train_batch_size: options.train_batch_size,
                epochs: options.epochs,
                learning_rate: options.learning_rate,
                policy_loss_weight: options.policy_loss_weight,
                value_loss_weight: options.value_loss_weight
            }
        ))
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
        let run_directory = SelfLearn::<S,A,E,M>::get_run_directory(game_name, run_name);
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
