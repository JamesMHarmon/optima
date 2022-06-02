use std::sync::Mutex;
use anyhow::Result;
use crossbeam::channel::Sender;
use futures::stream::{FuturesUnordered, StreamExt};
use log::info;
use serde::Serialize;
use std::fmt::Debug;
use std::sync::Arc;
use std::time::{Instant, Duration};

use engine::engine::GameEngine;
use engine::game_state::GameState;
use engine::value::Value;
use model::analytics::GameAnalyzer;
use model::model::{Model, ModelFactory, ModelArena};
use model::model_info::ModelInfo;
use super::{play_self_one, SelfPlayMetrics, SelfPlayOptions, SelfPlayPersistance};

pub fn play_self<F, M, MA, E, T, S, A, V>(
    model_factory: &F,
    model_arena: &MA,
    engine: &E,
    self_play_persistance: &mut SelfPlayPersistance,
    self_play_options: &SelfPlayOptions,
) -> Result<()>
where
    F: ModelFactory<M = M> + Sync,
    M: Model<State = S, Action = A, Analyzer = T, Value = V> + Send + Sync,
    MA: ModelArena + Sync,
    E: GameEngine<State = S, Action = A, Value = V> + Sync,
    T: GameAnalyzer<Action = A, State = S, Value = V> + Send,
    S: GameState + Send,
    A: Serialize + Debug + Eq + Clone + Send,
    V: Value + Serialize + Debug + Send,
{
    let starting_run_time = Instant::now();
    let (game_results_tx, game_results_rx) = crossbeam::channel::unbounded();
    let runtime_handle = tokio::runtime::Handle::current();

    let latest_model_info = model_arena.latest_certified()?;
    let latest_model = model_factory.get(&latest_model_info);
    let latest_model: Arc<Mutex<M>> = Arc::new(Mutex::new(latest_model));

    crossbeam::scope(move |s| {
        for thread_num in 0..self_play_options.self_play_parallelism {
            let game_results_tx = game_results_tx.clone();
            let runtime_handle = runtime_handle.clone();
            let latest_model = latest_model.clone();

            s.spawn(move |_| {
                info!("Starting Thread: {}", thread_num);
                let latest_model_analyzer = || {
                    let latest_model = latest_model.lock().unwrap();
                    (latest_model.get_game_state_analyzer(), latest_model.get_model_info().clone())
                };

                let f = play_games(
                    self_play_options.self_play_batch_size,
                    game_results_tx,
                    engine,
                    latest_model_analyzer,
                    &self_play_options
                );

                let res = runtime_handle.block_on(f);

                res.unwrap();
            });
        }

        s.spawn(move |_| {
            loop {
                let new_latest_model = model_arena.latest_certified().unwrap();
                {
                    let mut latest_model = latest_model.lock().unwrap();
                    let latest_model_info = latest_model.get_model_info();
                    if new_latest_model.get_model_name() != latest_model_info.get_model_name() || new_latest_model.get_model_num() != latest_model_info.get_model_num() {
                        let new_latest_model = model_factory.get(&new_latest_model);
                        
                        info!("Updating latest model from {:?} to {:?}", latest_model_info, new_latest_model.get_model_info());

                        *latest_model = new_latest_model;
                    }
                }

                std::thread::sleep(Duration::from_secs(10));
            }
        });

        s.spawn(move |_| -> Result<()> {
            let mut num_of_games_played: usize = 0;

            while let Ok((self_play_metric, game_state, model_info)) = game_results_rx.recv() {
                self_play_persistance.write(&self_play_metric, model_info).unwrap();
                num_of_games_played += 1;

                info!(
                    "Number of Actions: {}, Score: {:?}, Move: {:?}, Elapsed: {:.2}h, Number of Games Played: {}, GPM: {:.2}",
                    self_play_metric.analysis().len(),
                    self_play_metric.score(),
                    engine.get_move_number(&game_state),
                    starting_run_time.elapsed().as_secs() as f32 / (60 * 60) as f32,
                    num_of_games_played,
                    num_of_games_played as f32 / starting_run_time.elapsed().as_secs() as f32 * 60_f32
                );
            }

            Ok(())
        });
    }).unwrap();

    Ok(())
}

async fn play_games<M, E, S, A, V, F: Fn() -> (M, ModelInfo)>(
    self_play_batch_size: usize,
    results_channel: Sender<(SelfPlayMetrics<A, V>, S, ModelInfo)>,
    game_engine: &E,
    latest_model_analyzer: F,
    self_play_options: &SelfPlayOptions,
) -> Result<()>
where
    M: GameAnalyzer<Action = A, State = S, Value = V> + Send,
    E: GameEngine<State = S, Action = A, Value = V>,
    S: GameState,
    A: Clone + Eq + Debug,
    V: Value,
{
    let mut self_play_metric_stream = FuturesUnordered::new();

    let play_game = || async {
        let (analyzer, info) = latest_model_analyzer();
        let results = play_self_one(game_engine, &analyzer, self_play_options)
            .await
            .unwrap();
        (results.0, results.1, info)
    };

    for _ in 0..self_play_batch_size {
        self_play_metric_stream.push(play_game());
    }

    while let Some(self_play_metric) = self_play_metric_stream.next().await {
        results_channel
            .send(self_play_metric)
            .expect("Failed to send game result");

        self_play_metric_stream
        .push(play_game());
    }

    Ok(())
}
