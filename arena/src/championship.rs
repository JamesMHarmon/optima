use anyhow::{anyhow, Context, Result};
use engine::{GameEngine, GameState};
use log::{error, info, warn};
use model::{Analyzer, GameAnalyzer, Info, Latest, Load, Move};
use serde::{de::DeserializeOwned, Serialize};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::{fmt::Debug, time::Duration};
use tokio::runtime::Handle;

use super::ArenaOptions;
use super::{evaluate::EvalResult, EvaluatePersistance};

#[allow(clippy::too_many_arguments)]
pub fn championship<S, A, F, E, M, MR, T>(
    champions: &F,
    champions_dir: &Path,
    candidates: &F,
    certified_dir: &Path,
    evaluated_dir: &Path,
    engine: &E,
    run_dir: &Path,
    options: &ArenaOptions,
) -> Result<()>
where
    S: GameState,
    A: Clone + Eq + DeserializeOwned + Serialize + Debug + Unpin + Send + Sync + 'static,
    E: GameEngine<State = S, Action = A> + Sync,
    F: Load<MR = MR, M = M> + Latest<MR = MR> + Move<MR = MR> + Send + Sync,
    M: Analyzer<State = S, Action = A, Analyzer = T, Value = E::Value> + Info + Send + Sync,
    T: GameAnalyzer<Action = A, State = S, Value = E::Value> + Send,
    MR: Clone + Debug + Eq + Send + Sync,
{
    let runtime_handle = Handle::current();

    crossbeam::scope(|s| {
        let eval_candidates = Arc::new(Mutex::new(vec![]));

        loop {
            info!("Checking for latest candidate");

            let candidate = candidates.latest();

            if let Ok(candidate) = candidate {
                let mut eval_candidates_lock = eval_candidates.lock().unwrap();
                if !eval_candidates_lock.contains(&candidate) {

                    if eval_candidates_lock.len() >= 4 {
                        warn!("4 simultanious games are already in progress. Waiting for completion first.");
                        std::thread::sleep(Duration::from_secs(15));
                        continue;
                    }

                    eval_candidates_lock.push(candidate.clone());

                    drop(eval_candidates_lock);

                    let runtime_handle = runtime_handle.clone();

                    let eval_candidates = eval_candidates.clone();
                    s.spawn(move |_| {
                        runtime_handle.block_on(async {
                            let res = championship_single(
                                &candidate,
                                champions,
                                champions_dir,
                                candidates,
                                certified_dir,
                                evaluated_dir,
                                engine,
                                run_dir,
                                options,
                            );

                            if let Err(err) = res {
                                error!("Failed running single match: {:?}", err);
                            }

                            let mut eval_candidates_lock = eval_candidates.lock().unwrap();

                            eval_candidates_lock.retain(|c| c != &candidate);
                        });
                    });
                }
            } else {
                info!("No new candidates found");
                std::thread::sleep(Duration::from_secs(10));
            }

            std::thread::sleep(Duration::from_secs(5));
        }
    })
    .map_err(|e| {
        error!("{:?}", e);
        anyhow!("Failed in scope 1")
    })
    .and_then(|r| r)?;

    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn championship_single<S, A, F, E, M, MR, T>(
    candidate: &MR,
    champions: &F,
    champions_dir: &Path,
    candidates: &F,
    certified_dir: &Path,
    evaluated_dir: &Path,
    engine: &E,
    run_dir: &Path,
    options: &ArenaOptions,
) -> Result<()>
where
    S: GameState,
    A: Clone + Eq + DeserializeOwned + Serialize + Debug + Unpin + Send + Sync + 'static,
    E: GameEngine<State = S, Action = A> + Sync,
    F: Load<MR = MR, M = M> + Latest<MR = MR> + Move<MR = MR> + Send + Sync,
    M: Analyzer<State = S, Action = A, Analyzer = T, Value = E::Value> + Info + Send + Sync,
    T: GameAnalyzer<Action = A, State = S, Value = E::Value> + Send,
    MR: Debug + Send + Sync,
{
    let promote_candidate_to_champion = || {
        info!("Promoting {:?} to champion.", candidate);
        candidates.copy_to(candidate, champions_dir)
    };
    let promote_candidate_to_certified = || {
        info!("Certifying {:?}.", candidate);
        candidates.copy_to(candidate, certified_dir)
    };

    let champion = champions.latest();
    if let Ok(champion) = champion {
        let champion = champions
            .load(&champion)
            .with_context(|| "Failing to load current champion model")?;

        let candidate = candidates
            .load(candidate)
            .with_context(|| "Failing to load candidate model")?;

        let candidate_info = candidate.info().clone();

        let mut persistance = EvaluatePersistance::new(
            run_dir,
            &[champion.info().clone(), candidate.info().clone()],
        )?;

        info!(
            "Starting match between. {} and {}",
            champion.info().model_name_w_num(),
            candidate.info().model_name_w_num()
        );

        crossbeam::scope(|s| {
            let (tx, rx) = crossbeam::channel::unbounded();

            s.spawn(move |_| -> Result<()> {
                let mut candidate_score = 0.0f32;
                let mut has_certified = false;
                let mut has_championed = false;
                while let Ok(eval) = rx.recv() {
                    match eval {
                        EvalResult::GameResult(game_result) => {
                            candidate_score += game_result.model_score(&candidate_info);
                            if !has_certified && candidate_score >= options.certification_threshold
                            {
                                has_certified = true;
                                promote_candidate_to_certified()?;
                            }

                            if !has_championed && candidate_score >= options.champion_threshold {
                                has_championed = true;
                                promote_candidate_to_champion()?;
                            }

                            persistance.write_game(&game_result)?
                        }
                        EvalResult::MatchResult(match_result) => {
                            persistance.write_match(&match_result)?
                        }
                    }
                }

                Ok(())
            });

            super::evaluate::Arena::evaluate(&[champion, candidate], engine, tx, options)
        })
        .map_err(|e| {
            error!("{:?}", e);
            anyhow!("Failed in scope 1")
        })
        .and_then(|v| v)?;
    } else {
        info!("No champion found. Automatically promoting {:?}", &champion);
        promote_candidate_to_certified()?;
        promote_candidate_to_champion()?;
    }

    candidates.move_to(candidate, evaluated_dir)?;

    Ok(())
}
