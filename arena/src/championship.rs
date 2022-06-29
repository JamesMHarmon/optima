use anyhow::{anyhow, Result};
use engine::{GameEngine, GameState};
use log::{error, info};
use model::{Analyzer, GameAnalyzer, Info, Latest, Load};
use serde::{de::DeserializeOwned, Serialize};
use std::path::Path;
use std::{fmt::Debug, time::Duration};

use super::ArenaOptions;
use super::{evaluate::EvalResult, EvaluatePersistance};

pub fn championship<S, A, F, E, M, MR, T>(
    champions: &F,
    candidates: &F,
    engine: &E,
    run_dir: &Path,
    options: &ArenaOptions,
) -> Result<()>
where
    S: GameState,
    A: Clone + Eq + DeserializeOwned + Serialize + Debug + Unpin + Send + Sync + 'static,
    E: GameEngine<State = S, Action = A> + Sync,
    F: Load<MR = MR, M = M> + Latest<MR = MR>,
    M: Analyzer<State = S, Action = A, Analyzer = T, Value = E::Value> + Info + Send + Sync,
    T: GameAnalyzer<Action = A, State = S, Value = E::Value> + Send,
{
    loop {
        let res: Result<usize> = try {
            let candidate = candidates.latest();

            if let Ok(candidate) = candidate {
                let promote_candidate_to_champion = || {};
                let promote_candidate_to_certified = || {};

                let champion = champions.latest();
                if let Ok(champion) = champion {
                    let champion = champions.load(&champion)?;
                    let candidate = candidates.load(&candidate)?;

                    let mut persistance = EvaluatePersistance::new(
                        run_dir,
                        &[champion.info().clone(), candidate.info().clone()],
                    )?;

                    info!(
                        "Starting match between. {} and {}",
                        champion.info().model_name_w_num(),
                        candidate.info().model_name_w_num()
                    );

                    crossbeam::scope(|s| try {
                        let (tx, rx) = crossbeam::channel::unbounded();

                        s.spawn(move |_| {
                            let res: Result<usize> = try {
                                let mut candidate_score = 0.0f32;
                                let mut has_certified = false;
                                let mut has_championed = false;
                                while let Ok(eval) = rx.recv() {
                                    match eval {
                                        EvalResult::GameResult(game_result) => {
                                            candidate_score += game_result.scores[0].1;
                                            if !has_certified
                                                && candidate_score
                                                    >= options.certification_threshold
                                            {
                                                has_certified = true;
                                                promote_candidate_to_certified();
                                            }

                                            if !has_championed
                                                && candidate_score >= options.champion_threshold
                                            {
                                                has_championed = true;
                                                promote_candidate_to_champion();
                                            }

                                            persistance.write_game(&game_result)?;
                                        }
                                        EvalResult::MatchResult(match_result) => {
                                            persistance.write_match(&match_result)?;
                                        }
                                    }
                                }

                                0
                            };

                            res
                        });

                        super::evaluate::Arena::evaluate(
                            &[champion, candidate],
                            engine,
                            tx,
                            options,
                        )?;
                    })
                    .map_err(|e| {
                        error!("{:?}", e);
                        anyhow!("Failed in scope 1")
                    })
                    .flatten()?;
                } else {
                    info!("No champion found");
                    promote_candidate_to_champion();
                }
            } else {
                info!("No candidate found");
                std::thread::sleep(Duration::from_secs(15));
            }

            0
        };

        if let Err(err) = res {
            error!("Failed to run a championship. {:?}", err);
            std::thread::sleep(Duration::from_secs(15));
        }
    }
}
