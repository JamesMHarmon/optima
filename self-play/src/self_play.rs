use anyhow::Result;
use common::get_env_usize;
use common::{GameLength, PlayerToMove, TranspositionHash};
use futures::stream::{FuturesUnordered, StreamExt};
use log::info;
use log::warn;
use serde::Serialize;
use std::fmt::Debug;
use std::fmt::Display;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::{Duration, Instant};
use tokio::sync::mpsc::Sender;

use super::{SelfPlayMetrics, SelfPlayOptions, SelfPlayPersistance, play_self_one};
use engine::{GameEngine, GameState};
use mcts::SnapshotToPropagated;
use model::GameAnalyzer;
use model::{Analyzer, Info, Latest, Load, ModelInfo};
use puct::{RollupStats, SelectionPolicy, ValueModel};

type SnapshotOf<VM> = <<VM as ValueModel>::Rollup as RollupStats>::Snapshot;
type PVOf<VM> = <SnapshotOf<VM> as SnapshotToPropagated>::PropagatedValues;

pub fn play_self<F, M, E, S, A, P, VM, Sel>(
    model_factory: &F,
    engine: &E,
    value_model: &VM,
    selection_policy: &Sel,
    self_play_persistance: &mut SelfPlayPersistance,
    self_play_options: &SelfPlayOptions,
) -> Result<()>
where
    F: Latest + Load<MR = <F as Latest>::MR> + Load<M = M> + Sync,
    M: Analyzer<State = S, Action = A, Predictions = P> + Info + Send + Sync,
    M::Analyzer: GameAnalyzer<Action = A, State = S, Predictions = P> + Send,
    E: GameEngine<State = S, Action = A, Terminal = P> + Sync,
    P: Clone + engine::Value + GameLength,
    S: GameState + Clone + TranspositionHash + PlayerToMove + Send,
    A: Serialize + Debug + Eq + Clone + Send,
    P: Serialize + Display + Send,
    <F as Latest>::MR: Debug + Eq + Send,
    VM: ValueModel<State = S, Predictions = P, Terminal = P> + Send + Sync,
    Sel: SelectionPolicy<SnapshotOf<VM>, State = S> + Send + Sync,
    SnapshotOf<VM>: Clone + SnapshotToPropagated + Send + Sync,
    PVOf<VM>: Serialize + Send,
{
    let starting_run_time = Instant::now();
    let writer_channel_size = get_env_usize("WRITER_CHANNEL_SIZE").unwrap_or(1000);
    let (game_results_tx, mut game_results_rx) = tokio::sync::mpsc::channel(writer_channel_size);
    let runtime_handle = tokio::runtime::Handle::current();

    let latest_model_ref = model_factory.latest()?;
    let latest_model = model_factory.load(&latest_model_ref).unwrap();
    let latest_model: Arc<Mutex<(M, _)>> = Arc::new(Mutex::new((latest_model, latest_model_ref)));

    crossbeam::scope(move |s| {
        for thread_num in 0..self_play_options.self_play_parallelism {
            let game_results_tx = game_results_tx.clone();
            let runtime_handle = runtime_handle.clone();
            let latest_model = latest_model.clone();
            let value_model = value_model;
            let selection_policy = selection_policy;

            s.spawn(move |_| {
                info!("Starting Thread: {}", thread_num);
                let latest_model_analyzer = || {
                    let latest_model = latest_model.lock().unwrap();
                    (latest_model.0.analyzer(), latest_model.0.info().clone())
                };

                let f = play_games(
                    self_play_options.self_play_batch_size,
                    game_results_tx,
                    engine,
                    latest_model_analyzer,
                    value_model,
                    selection_policy,
                    self_play_options
                );

                let res = runtime_handle.block_on(f);

                res.unwrap();
            });
        }

        s.spawn(move |_| {
            loop {
                let new_latest_model_ref = model_factory.latest().unwrap();
                {
                    let mut latest_model = latest_model.lock().unwrap();

                    if new_latest_model_ref != latest_model.1 {
                        let new_latest_model = model_factory.load(&new_latest_model_ref).unwrap();

                        info!("Updating latest model from {:?} to {:?}", latest_model.1, new_latest_model.info());

                        *latest_model = (new_latest_model, new_latest_model_ref);
                    }
                }

                std::thread::sleep(Duration::from_secs(10));
            }
        });

        s.spawn(move |_| -> Result<()> {
            let mut num_of_games_played: usize = 0;

            while let Some((self_play_metric, game_state, model_info)) = game_results_rx.blocking_recv() {
                self_play_persistance.write(&self_play_metric, &model_info).unwrap();
                num_of_games_played += 1;

                info!(
                    "Model: {}, Number of Actions: {}, Score: {}, Move: {}, Transposition: {:X}, Elapsed: {:.2}h, Number of Games Played: {}, GPM: {:.2}",
                    model_info.model_name_w_num(),
                    self_play_metric.analysis().len(),
                    self_play_metric.terminal_score(),
                    engine.move_number(&game_state),
                    game_state.transposition_hash(),
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

async fn play_games<M, E, S, A, P, F, VM, Sel>(
    self_play_batch_size: usize,
    results_channel: Sender<(SelfPlayMetrics<A, P, PVOf<VM>>, S, ModelInfo)>,
    game_engine: &E,
    latest_model_analyzer: F,
    value_model: &VM,
    selection_policy: &Sel,
    self_play_options: &SelfPlayOptions,
) -> Result<()>
where
    F: Fn() -> (M, ModelInfo),
    S: GameState + Clone + TranspositionHash + PlayerToMove,
    A: Clone + Eq + Debug,
    E: GameEngine<State = S, Action = A, Terminal = P>,
    M: GameAnalyzer<State = S, Action = A, Predictions = P>,
    P: Clone + engine::Value + GameLength,
    VM: ValueModel<State = S, Predictions = P, Terminal = P> + Sync,
    Sel: SelectionPolicy<SnapshotOf<VM>, State = S> + Sync,
    SnapshotOf<VM>: Clone + SnapshotToPropagated,
    PVOf<VM>: Serialize + Send,
{
    let mut self_play_metric_stream = FuturesUnordered::new();

    let play_game = || async {
        let (analyzer, info) = latest_model_analyzer();
        let results = play_self_one(
            game_engine,
            &analyzer,
            value_model,
            selection_policy,
            self_play_options,
        )
        .await
        .unwrap();
        (results.0, results.1, info)
    };

    for _ in 0..self_play_batch_size {
        self_play_metric_stream.push(play_game());
    }

    while let Some(self_play_metric) = self_play_metric_stream.next().await {
        let result = results_channel.send(self_play_metric).await;

        if let Err(e) = result {
            warn!("Failed to send game results through writer channel. {}", e);
        }

        self_play_metric_stream.push(play_game());
    }

    Ok(())
}
