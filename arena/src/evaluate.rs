use anyhow::{anyhow, Context, Result};
use crossbeam::channel::Sender;
use futures::stream::{FuturesUnordered, StreamExt};
use log::{error, info};
use permutohedron::Heap as Permute;
use serde::de::DeserializeOwned;
use serde::Serialize;
use std::fmt::Debug;
use std::iter::{repeat, repeat_with};
use std::sync::mpsc;
use std::time::Instant;
use tokio::runtime::Handle;

use engine::{GameEngine, GameState, Value};
use mcts::{BackpropagationStrategy, SelectionStrategy, TemperatureMaxMoves, MCTS};
use model::ModelInfo;
use model::{Analyzer, GameAnalyzer, Info};

use super::{ArenaOptions, EVALUATE_PARALLELISM};

pub struct Arena {}

pub enum EvalResult<A> {
    GameResult(GameResult<A>),
    MatchResult(MatchResult),
}

#[derive(Debug, Serialize)]
pub struct GameResult<A> {
    pub actions: Vec<A>,
    pub scores: Vec<(ModelInfo, f32)>,
}

impl<A> GameResult<A> {
    pub fn model_score(&self, model_info: &ModelInfo) -> f32 {
        self.scores.iter().find(|(m, _)| m == model_info).unwrap().1
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct MatchResult {
    pub model_scores: Vec<(ModelInfo, f32)>,
    pub player_scores: Vec<f32>,
    pub num_of_games_played: usize,
}

impl Arena {
    pub fn evaluate<S, P, E, M, T, B, Sel, PV>(
        models: &[M],
        engine: &E,
        backpropagation_strategy: &B,
        selection_strategy: &Sel,
        results: Sender<EvalResult<E::Action>>,
        options: &ArenaOptions,
    ) -> Result<MatchResult>
    where
        S: GameState,
        E::Action: Clone + Eq + DeserializeOwned + Serialize + Debug + Unpin + Send + Sync + 'static,
        E: GameEngine<State = S, Terminal = P> + Sync,
        M: Analyzer<State = S, Action = E::Action, Analyzer = T, Predictions = P> + Info + Send + Sync,
        T: GameAnalyzer<Action = E::Action, State = S, Predictions = P> + Send,
        B: BackpropagationStrategy<State = S, Action = E::Action, Predictions = P, PropagatedValues = PV> + Send + Sync,
        Sel: SelectionStrategy<State = S, Action = E::Action, Predictions = P, PropagatedValues = PV> + Send + Sync,
        P: Value,
        PV: Default + Ord,
    {
        let num_players = models.len();
        let starting_time = Instant::now();

        let starting_run_time = Instant::now();
        let (game_results_tx, game_results_rx) = std::sync::mpsc::channel();

        let num_games_to_play = options.num_games;
        let batch_size = options.evaluate_batch_size;
        let runtime_handle = Handle::current();

        let match_result = crossbeam::scope(move |s| {
            let mut handles = vec![];
            let num_games_per_thread = num_games_to_play / EVALUATE_PARALLELISM;
            let num_games_per_thread_remainder = num_games_to_play % EVALUATE_PARALLELISM;

            for thread_num in 0..EVALUATE_PARALLELISM {
                let game_results_tx = game_results_tx.clone();
                let runtime_handle = runtime_handle.clone();

                let num_games_to_play_this_thread = num_games_per_thread
                    + if thread_num < num_games_per_thread_remainder {
                        1
                    } else {
                        0
                    };

                handles.push(s.spawn(move |_| {
                    info!(
                        "Starting Thread: {}, Games: {}",
                        thread_num, num_games_to_play_this_thread
                    );

                    let f = Self::play_games(
                        num_games_to_play_this_thread,
                        num_players,
                        batch_size,
                        game_results_tx,
                        engine,
                        models,
                        backpropagation_strategy,
                        selection_strategy,
                        options,
                    );

                    runtime_handle.block_on(f)
                }));
            }

            drop(game_results_tx);

            let model_info: Vec<_> = models.iter().map(|m| m.info().to_owned()).collect();

            let handle = s.spawn(move |_| -> Result<MatchResult> {
                let mut num_of_games_played: usize = 0;
                let mut model_scores: Vec<_> =
                    model_info.iter().map(|m| (m.to_owned(), 0.0)).collect();
                let mut player_scores: Vec<_> = repeat(0.0).take(num_players).collect();

                while let Ok(game_result) = game_results_rx.recv() {
                    num_of_games_played += 1;

                    for (i, (model_info, score)) in game_result.scores.iter().enumerate() {
                        player_scores[i] += score;
                        model_scores
                            .iter_mut()
                            .find(|(m, _s)| *m == *model_info)
                            .unwrap()
                            .1 += score;
                    }

                    info!("{:?}", game_result);

                    info!(
                        "Time Elapsed: {:.2}h, Number of Games Played: {}, GPM: {:.2}",
                        starting_time.elapsed().as_secs() as f32 / (60 * 60) as f32,
                        num_of_games_played,
                        num_of_games_played as f32 / starting_run_time.elapsed().as_secs() as f32
                            * 60_f32
                    );

                    info!("Model Scores: {:?}", model_scores);

                    results.send(EvalResult::GameResult(game_result))?;
                }

                let match_result = MatchResult {
                    num_of_games_played,
                    model_scores,
                    player_scores,
                };

                results.send(EvalResult::MatchResult(match_result.clone()))?;

                Ok(match_result)
            });

            for handle in handles {
                let res = handle
                    .join()?
                    .with_context(|| "Error in self_evaluate scope 1")
                    .map_err(|e| anyhow!("{:?}", e));

                if let Err(err) = res {
                    error!("{:?}", err);
                }
            }

            handle.join()
        });

        match_result
            .and_then(|v| v)
            .map_err(|e| {
                error!("{:?}", e);
                anyhow!("Error in self_evaluate scope 2")
            })
            .and_then(|v| v)
            .map_err(|e| {
                error!("{:?}", e);
                anyhow!("Error in self_evaluate scope 3")
            })
    }

    #[allow(clippy::too_many_arguments)]
    async fn play_games<S, A, P, E, M, T, B, Sel, PV>(
        num_games_to_play: usize,
        num_players: usize,
        batch_size: usize,
        results_channel: mpsc::Sender<GameResult<A>>,
        engine: &E,
        models: &[M],
        backpropagation_strategy: &B,
        selection_strategy: &Sel,
        options: &ArenaOptions,
    ) -> Result<()>
    where
        S: GameState,
        A: Clone + Eq + DeserializeOwned + Serialize + Debug + Unpin + Send,
        E: GameEngine<State = S, Action = A, Terminal = P> + Sync,
        M: Analyzer<State = S, Action = A, Analyzer = T, Predictions = P> + Info + Send + Sync,
        B: BackpropagationStrategy<State = S, Action = A, Predictions = P, PropagatedValues = PV> + Send + Sync,
        Sel: SelectionStrategy<State = S, Action = A, Predictions = P, PropagatedValues = PV> + Send + Sync,
        T: GameAnalyzer<State = S, Action = A, Predictions = P> + Send,
        P: Value,
        PV: Default + Ord,
    {
        let mut games_to_play_futures = FuturesUnordered::new();
        let repeated_models: Vec<_> = repeat_with(|| models.iter())
            .flatten()
            .take(num_players)
            .collect();

        let mut games_to_play: Vec<Vec<&M>> = repeat_with(|| repeated_models.clone())
            .flat_map::<Vec<_>, _>(|mut d| Permute::new(&mut d).collect())
            .take(num_games_to_play)
            .collect();

        for _ in 0..batch_size {
            if let Some(players) = games_to_play.pop() {
                let game_to_play = Self::play_game(engine, backpropagation_strategy,  selection_strategy, players, options);
                games_to_play_futures.push(game_to_play);
            }
        }

        while let Some(game_result) = games_to_play_futures.next().await {
            let game_result = game_result?;

            results_channel
                .send(game_result)
                .map_err(|_| anyhow!("Failed to send game_result"))?;

            if let Some(players) = games_to_play.pop() {
                let game_to_play = Self::play_game(engine, backpropagation_strategy,  selection_strategy, players, options);
                games_to_play_futures.push(game_to_play);
            }
        }

        Ok(())
    }

    #[allow(non_snake_case)]
    async fn play_game<S, A, E, B, Sel, T, M, P, PV>(
        engine: &E,
        backpropagation_strategy: &B,
        selection_strategy: &Sel,
        players: Vec<&M>,
        options: &ArenaOptions,
    ) -> Result<GameResult<A>>
    where
        S: GameState,
        A: Clone + Eq + DeserializeOwned + Serialize + Debug + Unpin + Send,
        E: GameEngine<State = S, Action = A, Terminal = P> + Sync,
        M: Analyzer<State = S, Action = A, Analyzer = T, Predictions = P> + Info + Send + Sync,
        B: BackpropagationStrategy<State = S, Action = A, Predictions = P, PropagatedValues = PV> + Send + Sync,
        Sel: SelectionStrategy<State = S, Action = A, Predictions = P, PropagatedValues = PV> + Send + Sync,
        T: GameAnalyzer<State = S, Action = A, Predictions = P> + Send,
        P: Value,
        PV: Default + Ord,
    {
        let play_options = &options.play_options;
        let visits = options.visits;
        let analyzers = players.iter().map(|m| m.analyzer()).collect::<Vec<_>>();

        let mut mctss: Vec<_> = analyzers
            .iter()
            .map(|a| {

                let temp = TemperatureMaxMoves::new(
                    play_options.temperature,
                    play_options.temperature_post_max_moves,
                    play_options.temperature_max_moves,
                    engine,
                );

                MCTS::with_capacity(
                    S::initial(),
                    engine,
                    a,
                    backpropagation_strategy,
                    selection_strategy,
                    visits,
                    temp,
                    play_options.parallelism
                )
            })
            .collect();

        let mut actions: Vec<A> = Vec::new();
        let mut state: S = S::initial();

        while engine.terminal_state(&state).is_none() {
            let player_to_move = engine.player_to_move(&state);
            let player_to_move_mcts = &mut mctss[player_to_move - 1];
            player_to_move_mcts.search_visits(visits).await?;
            let action = player_to_move_mcts.select_action()?;

            for mcts in &mut mctss {
                mcts.advance_to_action(action.to_owned()).await?;
            }

            state = engine.take_action(&state, &action);

            actions.push(action);
        }

        let final_score = engine
            .terminal_state(&state)
            .ok_or_else(|| anyhow!("Expected a terminal state"))?;

        let scores: Vec<_> = players
            .iter()
            .enumerate()
            .map(|(i, m)| (m.info().clone(), final_score.get_value_for_player(i + 1)))
            .collect();

        Ok(GameResult { actions, scores })
    }
}
