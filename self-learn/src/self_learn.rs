use anyhow::Result;
use crossbeam::Sender;
use futures::stream::{FuturesUnordered, StreamExt};
use log::info;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::path::Path;
use std::time::Instant;

use engine::engine::GameEngine;
use engine::game_state::GameState;
use engine::value::Value;
use model::analytics::GameAnalyzer;
use model::model::{Model, ModelFactory};
use model::model_info::ModelInfo;
use self_evaluate::self_evaluate::{SelfEvaluate, SelfEvaluateOptions};

use super::constants::SELF_PLAY_PARALLELISM;
use super::self_play::{self, SelfPlayMetrics, SelfPlayOptions};
use super::self_play_persistance::SelfPlayPersistance;
use super::train;

pub struct SelfLearn<'a, S, A, V, E>
where
    S: GameState,
    A: Clone + Eq + Serialize + Unpin,
    V: Value,
    E: 'a + GameEngine<State = S, Action = A, Value = V>,
{
    self_learn_options: &'a SelfLearnOptions,
    self_evaluate_options: &'a SelfEvaluateOptions,
    run_directory: &'a Path,
    game_engine: &'a E,
    model_info: ModelInfo,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct SelfLearnOptions {
    pub number_of_games_per_net: usize,
    pub self_play_batch_size: usize,
    pub parallelism: usize,
    pub moving_window_size: usize,
    pub max_moving_window_percentage: f32,
    pub position_sample_percentage: f32,
    pub train_ratio: f32,
    pub train_batch_size: usize,
    pub epochs: usize,
    pub learning_rate: f32,
    pub max_grad_norm: f32,
    pub policy_loss_weight: f32,
    pub value_loss_weight: f32,
    pub moves_left_loss_weight: f32,
    pub temperature: f32,
    pub temperature_max_actions: usize,
    pub temperature_post_max_actions: f32,
    pub temperature_visit_offset: f32,
    pub visits: usize,
    pub fast_visits: usize,
    pub full_visits_probability: f32,
    pub fpu: f32,
    pub fpu_root: f32,
    pub logit_q: bool,
    pub cpuct_base: f32,
    pub cpuct_init: f32,
    pub cpuct_root_scaling: f32,
    pub moves_left_threshold: f32,
    pub moves_left_scale: f32,
    pub moves_left_factor: f32,
    pub epsilon: f32,
}

impl<'a, S, A, V, E> SelfLearn<'a, S, A, V, E>
where
    S: GameState,
    A: Clone + Eq + DeserializeOwned + Serialize + Debug + Unpin + Send,
    V: Value + DeserializeOwned + Serialize,
    E: 'a + GameEngine<State = S, Action = A, Value = V> + Sync,
{
    pub fn create<M, T, F, O>(
        game_name: String,
        run_name: String,
        model_factory: &F,
        options: &O,
    ) -> Result<()>
    where
        M: Model<Action = A, State = S, Analyzer = T>,
        T: GameAnalyzer<Action = A, State = S, Value = M::Value> + Send,
        F: ModelFactory<M = M, O = O>,
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
    ) -> Result<Self> {
        let model_info = ModelInfo::new(game_name, run_name, 1);

        Ok(Self {
            self_learn_options,
            self_evaluate_options,
            run_directory,
            game_engine,
            model_info,
        })
    }

    pub fn learn<M, T, F>(&mut self, model_factory: &F) -> Result<()>
    where
        M: Model<Action = A, State = S, Analyzer = T, Value = V> + Send + Sync,
        T: GameAnalyzer<Action = A, State = S, Value = V> + Send,
        F: ModelFactory<M = M>,
        V: Clone + Send + Debug,
    {
        let self_learn_options = self.self_learn_options;
        let self_evaluate_options = self.self_evaluate_options;
        let run_directory = &self.run_directory;
        let game_engine = self.game_engine;
        let mut latest_model_info = model_factory.get_latest(&self.model_info)?;

        loop {
            let model_name = latest_model_info.get_model_name();

            let num_games_to_play = {
                let self_play_persistance =
                    SelfPlayPersistance::new(run_directory, model_name.to_owned())?;

                let num_games_this_net = self_play_persistance.read::<A, V>()?.count();
                let number_of_games_per_net = self_learn_options.number_of_games_per_net;
                let num_games_to_play = if num_games_this_net < number_of_games_per_net {
                    number_of_games_per_net - num_games_this_net
                } else {
                    0
                };
                drop(self_play_persistance);
                num_games_to_play
            };

            let mut self_play_persistance = SelfPlayPersistance::new(run_directory, model_name)?;

            let latest_model = model_factory.get(&latest_model_info);
            Self::play_self(
                &latest_model,
                game_engine,
                num_games_to_play,
                &mut self_play_persistance,
                self_learn_options,
            )?;

            let new_model_info = train::train_model::<S, A, V, E, M, T>(
                &latest_model,
                &self_play_persistance,
                game_engine,
                self_learn_options,
            )?;

            let new_model = model_factory.get(&new_model_info);
            SelfEvaluate::evaluate(
                &[latest_model, new_model],
                game_engine,
                self_evaluate_options,
            )?;

            latest_model_info = new_model_info;
        }
    }

    fn play_self<M, T>(
        model: &M,
        engine: &E,
        num_games_to_play: usize,
        self_play_persistance: &mut SelfPlayPersistance,
        options: &SelfLearnOptions,
    ) -> Result<()>
    where
        M: Model<State = S, Action = A, Analyzer = T, Value = V> + Send + Sync,
        T: GameAnalyzer<Action = A, State = S, Value = V> + Send,
        V: Send + Debug,
    {
        let self_play_batch_size = options.self_play_batch_size;
        let starting_run_time = Instant::now();
        let mut num_games_to_play = num_games_to_play;

        let (game_results_tx, game_results_rx) = crossbeam::channel::unbounded();

        let runtime_handle = tokio::runtime::Handle::current();

        crossbeam::scope(move |s| {
            let num_games_per_thread = num_games_to_play / SELF_PLAY_PARALLELISM;
            let num_games_per_thread_remainder = num_games_to_play % SELF_PLAY_PARALLELISM;

            for thread_num in 0..SELF_PLAY_PARALLELISM {
                let game_results_tx = game_results_tx.clone();
                let runtime_handle = runtime_handle.clone();
                let num_games_to_play_this_thread = num_games_per_thread + if thread_num == 0 { num_games_per_thread_remainder } else { 0 };

                let self_play_options = SelfPlayOptions {
                    epsilon: options.epsilon,
                    fpu: options.fpu,
                    fpu_root: options.fpu_root,
                    logit_q: options.logit_q,
                    cpuct_base: options.cpuct_base,
                    cpuct_init: options.cpuct_init,
                    cpuct_root_scaling: options.cpuct_root_scaling,
                    moves_left_threshold: options.moves_left_threshold,
                    moves_left_scale: options.moves_left_scale,
                    moves_left_factor: options.moves_left_factor,
                    temperature: options.temperature,
                    temperature_max_actions: options.temperature_max_actions,
                    temperature_post_max_actions: options.temperature_post_max_actions,
                    temperature_visit_offset: options.temperature_visit_offset,
                    visits: options.visits,
                    fast_visits: options.fast_visits,
                    full_visits_probability: options.full_visits_probability,
                    parallelism: options.parallelism
                };

                s.spawn(move |_| {
                    info!("Starting Thread: {}", thread_num);
                    let f = Self::play_games(
                        num_games_to_play_this_thread,
                        self_play_batch_size,
                        game_results_tx,
                        engine,
                        model,
                        &self_play_options
                    );

                    let res = runtime_handle.block_on(f);

                    res.unwrap();
                });
            }

            s.spawn(move |_| -> Result<()> {
                let mut num_of_games_played: usize = 0;

                while let Ok(self_play_metric) = game_results_rx.recv() {
                    self_play_persistance.write(&self_play_metric).unwrap();
                    num_of_games_played += 1;
                    num_games_to_play -= 1;

                    info!(
                        "Time Elapsed: {:.2}h, Number of Games Played: {}, Number of Games To Play: {}, GPM: {:.2}",
                        starting_run_time.elapsed().as_secs() as f32 / (60 * 60) as f32,
                        num_of_games_played,
                        num_games_to_play,
                        num_of_games_played as f32 / starting_run_time.elapsed().as_secs() as f32 * 60_f32
                    );

                    info!("Number of Actions: {}, Score: {:?}",
                        self_play_metric.get_analysis().len(),
                        self_play_metric.get_score()
                    );
                }

                Ok(())
            });
        }).unwrap();

        Ok(())
    }

    async fn play_games<M>(
        num_games_to_play: usize,
        self_play_batch_size: usize,
        results_channel: Sender<SelfPlayMetrics<A, V>>,
        game_engine: &E,
        model: &M,
        self_play_options: &SelfPlayOptions,
    ) -> Result<()>
    where
        M: Model<Action = A, State = S, Value = V> + Send,
    {
        let mut num_games_to_play = num_games_to_play;
        let mut self_play_metric_stream = FuturesUnordered::new();

        info!("To Play: {}", num_games_to_play);

        for _ in 0..std::cmp::min(num_games_to_play, self_play_batch_size) {
            self_play_metric_stream.push(self_play::self_play(
                game_engine,
                model,
                &self_play_options,
            ));
        }

        while let Some(self_play_metric) = self_play_metric_stream.next().await {
            let self_play_metric = self_play_metric.unwrap();
            num_games_to_play -= 1;

            results_channel
                .send(self_play_metric)
                .expect("Failed to send game result");

            if num_games_to_play - self_play_metric_stream.len() > 0 {
                self_play_metric_stream.push(self_play::self_play(
                    game_engine,
                    model,
                    &self_play_options,
                ));
            }
        }

        Ok(())
    }
}
