use std::path::PathBuf;
use std::fmt::Debug;
use std::time::Instant;
use std::sync::mpsc;
use serde::{Deserialize,Serialize};
use serde::de::{DeserializeOwned};
use futures::stream::{FuturesUnordered,StreamExt};
use futures::future::FutureExt;
use failure::{Error,format_err};
use tokio_executor::current_thread;

use mcts::mcts::{MCTS,MCTSOptions};
use model::analytics::GameAnalyzer;
use engine::engine::GameEngine;
use engine::game_state::GameState;
use model::model::{Model};
use model::model_info::{ModelInfo};

use super::self_evaluate_persistance::SelfEvaluatePersistance;
use super::constants::SELF_EVALUATE_PARALLELISM;

#[derive(Serialize, Deserialize, Debug)]
pub struct SelfEvaluateOptions {
    pub num_games: usize,
    pub batch_size: usize,
    pub parallelism: usize,
    pub temperature: f32,
    pub temperature_max_actions: usize,
    pub temperature_post_max_actions: f32,
    pub visits: usize,
    pub fpu: f32,
    pub fpu_root: f32,
    pub cpuct_base: f32,
    pub cpuct_init: f32,
    pub cpuct_root_scaling: f32
}

pub struct SelfEvaluate {}

#[derive(Debug,Serialize)]
pub struct GameResult<A> {
    model_1_num: usize,
    model_2_num: usize,
    actions: Vec<A>,
    score: f32
}

#[derive(Debug,Serialize)]
pub struct MatchResult {
    model_1_num: usize,
    model_2_num: usize,
    model_1_score: f32,
    model_2_score: f32,
    score_as_p1: f32,
    score_as_p2: f32,
    num_of_games_played: usize
}

impl SelfEvaluate
{
    pub fn evaluate<S,A,E,M,T>(
        model_1: &M,
        model_2: &M,
        game_engine: &E,
        options: &SelfEvaluateOptions
    ) -> Result<(), Error> 
    where
        S: GameState,
        A: Clone + Eq + DeserializeOwned + Serialize + Debug + Unpin + Send,
        E: GameEngine<State=S,Action=A> + Sync,
        M: Model<State=S,Action=A,Analyzer=T>,
        T: GameAnalyzer<Action=A,State=S> + Send
    {
        let model_1_info = model_1.get_model_info();
        let model_2_info = model_2.get_model_info();

        let model_1_num = model_1_info.get_model_num();
        let model_2_num = model_2_info.get_model_num();

        let starting_time = Instant::now();

        let starting_run_time = Instant::now();
        let (game_results_tx, game_results_rx) = std::sync::mpsc::channel();

        let num_games_to_play = options.num_games;
        let batch_size = options.batch_size;

        crossbeam::scope(move |s| {
            let num_games_per_thread = num_games_to_play / SELF_EVALUATE_PARALLELISM;
            let num_games_per_thread_remainder = num_games_to_play % SELF_EVALUATE_PARALLELISM;

            for thread_num in 0..SELF_EVALUATE_PARALLELISM {
                let game_results_tx = game_results_tx.clone();

                let analyzer_1 = model_1.get_game_state_analyzer();
                let analyzer_2 = model_2.get_game_state_analyzer();
                let num_games_to_play_this_thread = num_games_per_thread + if thread_num == 0 { num_games_per_thread_remainder } else { 0 };

                s.spawn(move |_| {
                    println!("Starting Thread: {}, Games: {}", thread_num, num_games_to_play_this_thread);

                    let f = Self::play_games(
                        num_games_to_play_this_thread,
                        batch_size,
                        game_results_tx,
                        game_engine,
                        (model_1_info, &analyzer_1),
                        (model_2_info, &analyzer_2),
                        options
                    ).map(|_| ());

                    current_thread::block_on_all(f);
                });
            }

            s.spawn(move |_| -> Result<(), Error> {
                let mut num_of_games_played: usize = 0;
                let mut model_1_score: f32 = 0.0;
                let mut model_2_score: f32 = 0.0;
                let mut score_as_p1: f32 = 0.0;
                let mut score_as_p2: f32 = 0.0;

                let mut presistance = SelfEvaluatePersistance::new(
                    &get_run_directory(model_1_info.get_game_name(), model_1_info.get_run_name()),
                    model_1_info,
                    model_2_info
                )?;

                while let Ok(game_result) = game_results_rx.recv() {
                    num_of_games_played += 1;

                    let normalized_score = (game_result.score + 1.0) / 2.0;

                    if model_1_num == game_result.model_1_num {
                        model_1_score += normalized_score;
                        model_2_score += 1.0 - normalized_score;
                    } else {
                        model_2_score += normalized_score;
                        model_1_score += 1.0 - normalized_score;
                    }

                    score_as_p1 += normalized_score;
                    score_as_p2 += 1.0 - normalized_score;

                    println!("{:?}", game_result);

                    println!(
                        "Time Elapsed: {:.2}h, Number of Games Played: {}, GPM: {:.2}",
                        starting_time.elapsed().as_secs() as f32 / (60 * 60) as f32,
                        num_of_games_played,
                        num_of_games_played as f32 / starting_run_time.elapsed().as_secs() as f32 * 60 as f32
                    );

                    presistance.write_game(&game_result)?;
                }

                presistance.write_match(&MatchResult {
                    num_of_games_played,
                    model_1_num,
                    model_2_num,
                    model_1_score,
                    model_2_score,
                    score_as_p1,
                    score_as_p2
                })?;

                Ok(())
            });
        }).unwrap();

        Ok(())
    }

    async fn play_games<S, A, E, T>(
        num_games_to_play: usize,
        batch_size: usize,
        results_channel: mpsc::Sender<GameResult<A>>,
        game_engine: &E,
        model_1: (&ModelInfo, &T),
        model_2: (&ModelInfo, &T),
        options: &SelfEvaluateOptions
    ) -> Result<(), Error>
    where
        S: GameState,
        A: Clone + Eq + DeserializeOwned + Serialize + Debug + Unpin + Send,
        E: GameEngine<State=S,Action=A> + Sync,
        T: GameAnalyzer<Action=A,State=S> + Send
    {
        let mut game_result_stream = FuturesUnordered::new();
        let mut games_to_play: Vec<_> = (0..num_games_to_play).map(|i| {
            let (p1, p2) = if i % 2 == 0 {
                (model_1, model_2)
            } else {
                (model_2, model_1)
            };

            (p1, p2)
        }).collect();

        for _ in 0..batch_size {
            if let Some((p1, p2)) = games_to_play.pop() {
                let game_to_play = Self::play_game(game_engine, p1, p2, options);
                game_result_stream.push(game_to_play);
            }
        }

        while let Some(game_result) = game_result_stream.next().await {
            let game_result = game_result?;

            results_channel.send(game_result).map_err(|_| format_err!("Failed to send game_result"))?;

            if let Some((p1, p2)) = games_to_play.pop() {
                let game_to_play = Self::play_game(game_engine, p1, p2, options);
                game_result_stream.push(game_to_play);
            }
        }

        Ok(())
    }

    #[allow(non_snake_case)]
    async fn play_game<S, A, E, T>(
        game_engine: &E,
        model_1: (&ModelInfo, &T),
        model_2: (&ModelInfo, &T),
        options: &SelfEvaluateOptions
    ) -> Result<GameResult<A>, Error>
    where
        S: GameState,
        A: Clone + Eq + DeserializeOwned + Serialize + Debug + Unpin + Send,
        E: GameEngine<State=S,Action=A> + Sync,
        T: GameAnalyzer<Action=A,State=S> + Send
    {
        let fpu = options.fpu;
        let fpu_root = options.fpu_root;
        let cpuct_base = options.cpuct_base;
        let cpuct_init = options.cpuct_init;
        let cpuct_root_scaling = options.cpuct_root_scaling;
        let temperature_max_actions = options.temperature_max_actions;
        let temperature = options.temperature;
        let temperature_post_max_actions = options.temperature_post_max_actions;
        let visits = options.visits;
        let mut p1_last_to_move = false;

        let mut mcts_1 = MCTS::with_capacity(
            S::initial(),
            0,
            game_engine,
            model_1.1,
            MCTSOptions::<S,A,_,_>::new(
                None,
                fpu,
                fpu_root,
                |_,_,_,Nsb,is_root| (((Nsb as f32 + cpuct_base + 1.0) / cpuct_base).ln() + cpuct_init) * if is_root { cpuct_root_scaling } else { 1.0 },
                |_,num_actions| if num_actions < temperature_max_actions { temperature } else { temperature_post_max_actions },
                options.parallelism
            ),
            visits
        );

        let mut mcts_2 = MCTS::with_capacity(
            S::initial(),
            0,
            game_engine,
            model_2.1,
            MCTSOptions::<S,A,_,_>::new(
                None,
                fpu,
                fpu_root,
                |_,_,_,Nsb,is_root| (((Nsb as f32 + cpuct_base + 1.0) / cpuct_base).ln() + cpuct_init) * if is_root { cpuct_root_scaling } else { 1.0 },
                |_,num_actions| if num_actions < temperature_max_actions { temperature } else { temperature_post_max_actions },
                options.parallelism
            ),
            visits
        );

        let mut actions: Vec<A> = Vec::new();
        let mut state: S = S::initial();

        while game_engine.is_terminal_state(&state) == None {
            let action = if !p1_last_to_move {
                mcts_1.search(visits).await?;
                mcts_1.select_action().await?
            } else {
                mcts_2.search(visits).await?;
                mcts_2.select_action().await?
            };

            mcts_1.advance_to_action(action.to_owned()).await?;
            mcts_2.advance_to_action(action.to_owned()).await?;

            state = game_engine.take_action(&state, &action);

            actions.push(action);

            p1_last_to_move = !p1_last_to_move;
        };

        let final_score = game_engine.is_terminal_state(&state).ok_or(format_err!("Expected a terminal state"))?;
        let score = if p1_last_to_move { final_score * -1.0 } else { final_score };

        Ok(GameResult {
            model_1_num: model_1.0.get_model_num(),
            model_2_num: model_2.0.get_model_num(),
            actions,
            score
        })
    }
}

fn get_run_directory(game_name: &str, run_name: &str) -> PathBuf {
    PathBuf::from(format!("./{}_runs/{}", game_name, run_name))
}
