use anyhow::{Result, anyhow};
use common::get_env_usize;
use common::{GameLength, PlayerToMove, TranspositionHash};
use log::info;
use log::warn;
use serde::Serialize;
use std::fmt::Debug;
use std::fmt::Display;
use std::hash::Hash;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use super::{SelfPlayOptions, SelfPlayPersistance, play_self_one};
use engine::{GameEngine, GameState, ValidActions};
use model::GameAnalyzer;
use model::{Analyzer, Info, Latest, Load};
use puct::{RollupStats, SelectionPolicy, ValueModel};

type SnapshotOf<VM> = <<VM as ValueModel>::Rollup as RollupStats>::Snapshot;

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
    M::Analyzer: GameAnalyzer<Action = A, State = S, Predictions = P> + Send + Sync,
    E: GameEngine<State = S, Action = A, Terminal = P> + ValidActions<State = S, Action = A> + Sync,
    P: Clone + GameLength,
    S: GameState + Clone + TranspositionHash + PlayerToMove + Send + Sync,
    A: Serialize + Debug + Eq + Hash + Clone + Send + Sync,
    P: Serialize + Display + Send,
    <F as Latest>::MR: Debug + Eq + Send,
    VM: ValueModel<Predictions = P, Terminal = P> + Send + Sync,
    Sel: SelectionPolicy<SnapshotOf<VM>, State = S, Action = A, Terminal = P> + Send + Sync,
    <VM as ValueModel>::Rollup: Send + Sync,
    SnapshotOf<VM>: Clone + Serialize + Send + Sync,
{
    let starting_run_time = Instant::now();
    let writer_channel_size = get_env_usize("WRITER_CHANNEL_SIZE").unwrap_or(1000);
    let (game_results_tx, game_results_rx) = crossbeam::channel::bounded(writer_channel_size);

    let latest_model_ref = model_factory.latest()?;
    let latest_model = model_factory.load(&latest_model_ref).unwrap();
    let latest_model: Arc<Mutex<(M, _)>> = Arc::new(Mutex::new((latest_model, latest_model_ref)));

    let total_game_threads = self_play_options.concurrent_games;

    crossbeam::scope(move |s| {
        for thread_num in 0..total_game_threads {
            let game_results_tx = game_results_tx.clone();
            let latest_model = latest_model.clone();
            let value_model = value_model;
            let selection_policy = selection_policy;

            s.spawn(move |_| {
                info!("Starting game thread: {}", thread_num);
                let latest_model_analyzer = || {
                    let latest_model = latest_model
                        .lock()
                        .unwrap_or_else(|e| e.into_inner());
                    (latest_model.0.analyzer(), latest_model.0.info().clone())
                };

                loop {
                    // @TODO: Clean this panic catch
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        let (analyzer, model_info) = latest_model_analyzer();
                        let result = play_self_one(
                            engine,
                            &analyzer,
                            value_model,
                            selection_policy,
                            self_play_options,
                        );
                        result.map(|(metrics, game_state)| (metrics, game_state, model_info))
                    }));

                    match result {
                        Ok(Ok((metrics, game_state, model_info))) => {
                            game_results_tx
                                .send((metrics, game_state, model_info))
                                .ok();
                        }
                        Ok(Err(e)) => warn!("Game thread {} error: {}", thread_num, e),
                        Err(_) => warn!("Game thread {} panicked, continuing", thread_num),
                    }
                }
            });
        }

        s.spawn(move |_| {
            loop {
                let new_latest_model_ref = model_factory.latest().unwrap();
                {
                    let mut latest_model = latest_model
                        .lock()
                        .unwrap_or_else(|e| e.into_inner());

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

            while let Ok((self_play_metric, game_state, model_info)) = game_results_rx.recv() {
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
    }).map_err(|_| anyhow!("A self-play thread panicked"))?;

    Ok(())
}
