use std::path::Path;
use std::fmt::Debug;
use std::time::Instant;
use std::sync::mpsc;
use serde::{Serialize, Deserialize};
use serde::de::DeserializeOwned;
use futures::stream::{FuturesUnordered,StreamExt};
use futures::future::FutureExt;
use failure::{Error};
use tokio_executor::current_thread;

use model::analytics::GameAnalyzer;
use engine::engine::GameEngine;
use engine::game_state::GameState;
use model::model::{Model, ModelFactory};
use model::model_info::ModelInfo;
use self_evaluate::self_evaluate::{SelfEvaluate,SelfEvaluateOptions};

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
    self_learn_options: &'a SelfLearnOptions,
    self_evaluate_options: &'a SelfEvaluateOptions,
    run_directory: &'a Path,
    game_engine: &'a E,
    model_info: ModelInfo
}

#[derive(Serialize, Deserialize, Debug)]
pub struct SelfLearnOptions {
    pub number_of_games_per_net: usize,
    pub self_play_batch_size: usize,
    pub parallelism: usize,
    pub moving_window_size: usize,
    pub max_moving_window_percentage: f32,
    pub position_sample_percentage: f32,
    pub exclude_drawn_games: bool,
    pub train_ratio: f32,
    pub train_batch_size: usize,
    pub epochs: usize,
    pub learning_rate: f32,
    pub policy_loss_weight: f32,
    pub value_loss_weight: f32,
    pub temperature: f32,
    pub temperature_max_actions: usize,
    pub temperature_post_max_actions: f32,
    pub visits: usize,
    pub fpu: f32,
    pub fpu_root: f32,
    pub cpuct_base: f32,
    pub cpuct_init: f32,
    pub cpuct_root_scaling: f32,
    pub alpha: f32,
    pub epsilon: f32
}

impl<'a,S,A,E> SelfLearn<'a,S,A,E>
where
    S: GameState,
    A: Clone + Eq + DeserializeOwned + Serialize + Debug + Unpin + Send,
    E: 'a + GameEngine<State=S,Action=A> + Sync
{
    pub fn create<M,T,F,O>(
        game_name: String,
        run_name: String,
        model_factory: &F,
        options: &O
    ) -> Result<(), Error>
    where
        M: Model<Action=A,State=S,Analyzer=T>,
        T: GameAnalyzer<Action=A,State=S> + Send,
        F: ModelFactory<M=M,O=O>
    {
        let model_info = ModelInfo::new(game_name, run_name, 1);

        model_factory.create(&model_info, options);

        Ok(())
    }

    pub fn from(
        game_name: String,
        run_name: String,
        game_engine: &'a E,
        run_directory: &'a Path,
        self_learn_options: &'a SelfLearnOptions,
        self_evaluate_options: &'a SelfEvaluateOptions,
    ) -> Result<Self, Error>
    {
        let model_info = ModelInfo::new(game_name, run_name, 1);

        Ok(Self {
            self_learn_options,
            self_evaluate_options,
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
        let self_learn_options = self.self_learn_options;
        let self_evaluate_options = self.self_evaluate_options;
        let run_directory = &self.run_directory;
        let game_engine = self.game_engine;
        let mut latest_model_info = model_factory.get_latest(&self.model_info)?;

        loop {
            let model_name = latest_model_info.get_model_name();

            let num_games_to_play = {
                let self_play_persistance = SelfPlayPersistance::new(
                    run_directory,
                    model_name.to_owned()
                )?;

                let num_games_this_net = self_play_persistance.read::<A>()?.len();
                let number_of_games_per_net = self_learn_options.number_of_games_per_net;
                let num_games_to_play = if num_games_this_net < number_of_games_per_net { number_of_games_per_net - num_games_this_net } else { 0 };
                drop(self_play_persistance);
                num_games_to_play
            };

            let mut self_play_persistance = SelfPlayPersistance::new(
                run_directory,
                model_name
            )?;

            let latest_model = model_factory.get(&latest_model_info);
            Self::play_self(
                &latest_model,
                game_engine,
                num_games_to_play,
                &mut self_play_persistance,
                self_learn_options
            )?;

            let new_model_info = train::train_model::<S,A,E,M,T>(
                &latest_model,
                &self_play_persistance,
                game_engine,
                self_learn_options
            )?;

            let new_model = model_factory.get(&new_model_info);
            SelfEvaluate::evaluate(
                &latest_model,
                &new_model,
                game_engine,
                self_evaluate_options
            )?;

            latest_model_info = new_model_info;

            drop(latest_model);
            drop(self_play_persistance);
        }
    }

    fn play_self<M,T>(
        model: &M,
        engine: &E,
        num_games_to_play: usize,
        self_play_persistance: &mut SelfPlayPersistance,
        options: &SelfLearnOptions,
    ) -> Result<(), Error>
    where
        M: Model<State=S,Action=A,Analyzer=T>,
        T: GameAnalyzer<Action=A,State=S> + Send
    {
        let self_play_batch_size = options.self_play_batch_size;
        let starting_run_time = Instant::now();
        let mut num_games_to_play = num_games_to_play;

        let (game_results_tx, game_results_rx) = std::sync::mpsc::channel();

        crossbeam::scope(move |s| {
            let num_games_per_thread = num_games_to_play / SELF_PLAY_PARALLELISM;
            let num_games_per_thread_remainder = num_games_to_play % SELF_PLAY_PARALLELISM;

            for thread_num in 0..SELF_PLAY_PARALLELISM {
                let game_results_tx = game_results_tx.clone();
                let analyzer = model.get_game_state_analyzer();
                let num_games_to_play_this_thread = num_games_per_thread + if thread_num == 0 { num_games_per_thread_remainder } else { 0 };

                let self_play_options = SelfPlayOptions {
                    epsilon: options.epsilon,
                    alpha: options.alpha,
                    fpu: options.fpu,
                    fpu_root: options.fpu_root,
                    cpuct_base: options.cpuct_base,
                    cpuct_init: options.cpuct_init,
                    cpuct_root_scaling: options.cpuct_root_scaling,
                    temperature: options.temperature,
                    temperature_max_actions: options.temperature_max_actions,
                    temperature_post_max_actions: options.temperature_post_max_actions,
                    visits: options.visits,
                    parallelism: options.parallelism
                };

                s.spawn(move |_| {
                    println!("Starting Thread: {}", thread_num);
                    let f = Self::play_games(
                        num_games_to_play_this_thread,
                        self_play_batch_size,
                        game_results_tx,
                        engine,
                        analyzer,
                        &self_play_options
                    ).map(|_| ());

                    current_thread::block_on_all(f);
                });
            }

            s.spawn(move |_| -> Result<(), Error> {
                let mut num_of_games_played: usize = 0;

                while let Ok(self_play_metric) = game_results_rx.recv() {
                    self_play_persistance.write(&self_play_metric).unwrap();
                    num_of_games_played += 1;
                    num_games_to_play -= 1;

                    println!(
                        "Time Elapsed: {:.2}h, Number of Games Played: {}, Number of Games To Play: {}, GPM: {:.2}",
                        starting_run_time.elapsed().as_secs() as f32 / (60 * 60) as f32,
                        num_of_games_played,
                        num_games_to_play,
                        num_of_games_played as f32 / starting_run_time.elapsed().as_secs() as f32 * 60 as f32
                    );
                }

                Ok(())
            });
        }).unwrap();

        Ok(())
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
}
