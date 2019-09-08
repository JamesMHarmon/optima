use std::fmt::Debug;
use std::time::Instant;
use std::io::Write;
use std::io::Read;
use std::fs::{create_dir_all, OpenOptions};
use std::path::{Path,PathBuf};
use std::sync::mpsc;
use serde::{Serialize, Deserialize};
use serde::de::DeserializeOwned;
use futures::stream::{FuturesUnordered,StreamExt};
use futures::future::FutureExt;
use failure::{Error,format_err};

use model::analytics::GameAnalyzer;
use engine::engine::GameEngine;
use engine::game_state::GameState;
use model::model::{Model, ModelFactory};
use model::model_info::ModelInfo;

use super::self_play::{self,SelfPlayOptions,SelfPlayMetrics};
use super::self_play_persistance::{SelfPlayPersistance};
use super::constants::SELF_PLAY_PARALLELISM;
use super::train;


pub struct SelfLearn<'a, S, A, E>
where
    S: GameState,
    A: Clone + Eq + Serialize + Unpin,
    E: 'a + GameEngine<State=S,Action=A>
{
    options: SelfLearnOptions,
    run_directory: PathBuf,
    game_engine: &'a E,
    model_info: ModelInfo
}

#[derive(Serialize, Deserialize, Debug)]
pub struct SelfLearnOptions {
    pub number_of_games_per_net: usize,
    pub self_play_batch_size: usize,
    pub moving_window_size: usize,
    pub max_moving_window_percentage: f64,
    pub position_sample_percentage: f64,
    pub train_ratio: f64,
    pub train_batch_size: usize,
    pub epochs: usize,
    pub learning_rate: f64,
    pub policy_loss_weight: f64,
    pub value_loss_weight: f64,
    pub temperature: f64,
    pub temperature_max_actions: usize,
    pub temperature_post_max_actions: f64,
    pub visits: usize,
    pub cpuct_base: f64,
    pub cpuct_init: f64,
    pub alpha: f64,
    pub epsilon: f64,
    pub number_of_filters: usize,
    pub number_of_residual_blocks: usize
}

impl<'a,S,A,E> SelfLearn<'a,S,A,E>
where
    S: GameState,
    A: Clone + Eq + DeserializeOwned + Serialize + Debug + Unpin + Send,
    E: 'a + GameEngine<State=S,Action=A> + Sync
{
    pub fn create<M,T,F>(
        game_name: String,
        run_name: String,
        model_factory: &F,
        options: &SelfLearnOptions
    ) -> Result<(), Error>
    where
        M: Model<Action=A,State=S,Analyzer=T>,
        T: GameAnalyzer<Action=A,State=S> + Send,
        F: ModelFactory<M=M>
    {
        if game_name.contains("_") {
            return Err(format_err!("game_name cannot contain any '_' characters"));
        }

        if run_name.contains("_") {
            return Err(format_err!("run_name cannot contain any '_' characters"));
        }

        SelfLearn::<S,A,E>::initialize_directories_and_files(&game_name, &run_name, &options)?;
        let model_info = ModelInfo::new(game_name, run_name, 1);

        model_factory.create(&model_info, options.number_of_filters, options.number_of_residual_blocks);

        Ok(())
    }

    pub fn from(
        game_name: String,
        run_name: String,
        game_engine: &'a E
    ) -> Result<Self, Error>
    {
        let run_directory = Self::get_run_directory(&game_name, &run_name);
        let options = Self::get_config(&run_directory)?;
        let model_info = ModelInfo::new(game_name, run_name, 1);

        Ok(Self {
            options,
            run_directory,
            game_engine,
            model_info
        })
    }

    pub fn learn<M,T,F>(&mut self, model_factory: &F) -> Result<(), Error>
    where
        M: Model<Action=A,State=S,Analyzer=T>,
        T: GameAnalyzer<Action=A,State=S> + Send,
        F: ModelFactory<M=M>
    {
        let options = &self.options;
        let run_directory = &self.run_directory;
        let number_of_games_per_net = options.number_of_games_per_net;
        let self_play_batch_size = options.self_play_batch_size;
        let starting_time = Instant::now();

        loop {
            let starting_run_time = Instant::now();
            let latest_model = &model_factory.get_latest(&self.model_info);
            let model_name = latest_model.get_model_info().get_model_name();
            let (game_results_tx, game_results_rx) = std::sync::mpsc::channel();

            let self_play_persistance = SelfPlayPersistance::new(
                &self.run_directory,
                model_name.to_owned()
            )?;

            let mut num_games_this_net = self_play_persistance.read::<A>()?.len();
            let mut num_games_to_play = if num_games_this_net < number_of_games_per_net { number_of_games_per_net - num_games_this_net } else { 0 };

            let game_engine = self.game_engine;

            crossbeam::scope(move |s| {
                let num_games_per_thread = num_games_to_play / SELF_PLAY_PARALLELISM;
                let num_games_per_thread_remainder = num_games_to_play % SELF_PLAY_PARALLELISM;

                for thread_num in 0..SELF_PLAY_PARALLELISM {
                    let game_results_tx = game_results_tx.clone();
                    let analyzer = latest_model.get_game_state_analyzer();
                    let num_games_to_play_this_thread = num_games_per_thread + if thread_num == 0 { num_games_per_thread_remainder } else { 0 };

                    let self_play_options = SelfPlayOptions {
                        epsilon: options.epsilon,
                        alpha: options.alpha,
                        cpuct_base: options.cpuct_base,
                        cpuct_init: options.cpuct_init,
                        temperature: options.temperature,
                        temperature_max_actions: options.temperature_max_actions,
                        temperature_post_max_actions: options.temperature_post_max_actions,
                        visits: options.visits
                    };

                    s.spawn(move |_| {
                        println!("Starting Thread: {}", thread_num);
                        let f = Self::play_games(
                            num_games_to_play_this_thread,
                            self_play_batch_size,
                            game_results_tx,
                            game_engine,
                            analyzer,
                            &self_play_options
                        ).map(|_| ());

                        tokio_current_thread::block_on_all(f);
                    });
                }

                s.spawn(move |_| -> Result<(), Error> {
                    let mut num_of_games_played: usize = 0;
                    let mut self_play_persistance = SelfPlayPersistance::new(
                        run_directory,
                        model_name
                    )?;

                    while let Ok(self_play_metric) = game_results_rx.recv() {
                        self_play_persistance.write(&self_play_metric).unwrap();
                        num_games_this_net += 1;
                        num_of_games_played += 1;
                        num_games_to_play -= 1;

                        println!(
                            "Time Elapsed: {:.2}h, Number of Games Played: {}, Number of Games To Play: {}, GPM: {:.2}",
                            starting_time.elapsed().as_secs() as f64 / (60 * 60) as f64,
                            num_of_games_played,
                            num_games_to_play,
                            num_of_games_played as f64 / starting_run_time.elapsed().as_secs() as f64 * 60 as f64
                        );
                    }

                    Ok(())
                });
            }).unwrap();

            train::train_model::<S,A,E,M,T>(
                latest_model,
                &self_play_persistance,
                &self.game_engine,
                options
            )?;
        }
    }

    async fn play_games<T>(
        num_games_to_play: usize,
        self_play_batch_size: usize,
        results_channel: mpsc::Sender<SelfPlayMetrics<A>>,
        game_engine: &E,
        analyzer: T,
        self_play_options: &SelfPlayOptions
    ) -> Result<(), Error>
    where
        T: GameAnalyzer<Action=A,State=S> + Send,
    {
        let mut num_games_to_play = num_games_to_play;
        let mut self_play_metric_stream = FuturesUnordered::new();

        println!("To Play: {}", num_games_to_play);

        for _ in 0..std::cmp::min(num_games_to_play, self_play_batch_size) {
            self_play_metric_stream.push(
                self_play::self_play(game_engine, &analyzer, &self_play_options)
            );
        }

        while let Some(self_play_metric) = self_play_metric_stream.next().await {
            let self_play_metric = self_play_metric.unwrap();
            num_games_to_play -= 1;

            results_channel.send(self_play_metric).expect("Failed to send game result");

            if num_games_to_play - self_play_metric_stream.len() > 0 {
                self_play_metric_stream.push(
                    self_play::self_play(game_engine, &analyzer, &self_play_options)
                );
            }
        }

        Ok(())
    }

    fn get_run_directory(game_name: &str, run_name: &str) -> PathBuf {
        PathBuf::from(format!("./{}_runs/{}", game_name, run_name))
    }

    fn get_config_path(run_directory: &Path) -> PathBuf {
        run_directory.join("config.json")
    }

    fn get_config(run_directory: &Path) -> Result<SelfLearnOptions, Error> {
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

    fn initialize_directories_and_files(game_name: &str, run_name: &str, options: &SelfLearnOptions) -> Result<PathBuf, Error> {
        let run_directory = SelfLearn::<S,A,E>::get_run_directory(game_name, run_name);
        create_dir_all(&run_directory).expect("Run already exists or unable to create directories");

        let config_path = Self::get_config_path(&run_directory);

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
}
