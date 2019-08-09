use std::fmt::Debug;
use std::time::Instant;
use std::sync::mpsc;
use serde::{Serialize};
use serde::de::DeserializeOwned;
use futures::stream::{FuturesUnordered,StreamExt};
use futures::future::FutureExt;
use uuid::Uuid;

use common::linked_list::List;
use common::rng;
use mcts::mcts::{MCTS,MCTSOptions};
use model::analytics::GameAnalyzer;
use engine::engine::GameEngine;
use engine::game_state::GameState;
use model::model::{Model,ModelFactory};
use model::model_info::{ModelInfo};

#[derive(Debug)]
pub struct SelfEvaluateOptions {
    pub temperature: f64,
    pub temperature_max_actions: usize,
    pub temperature_post_max_actions: f64,
    pub visits: usize,
    pub cpuct_base: f64,
    pub cpuct_init: f64
}

pub struct SelfEvaluate {}

#[derive(Debug)]
pub struct MatchResult<A> {
    guid: String,
    p1_model_num: usize,
    p2_model_num: usize,
    actions: Vec<A>,
    score: f64
}

impl SelfEvaluate
{
    pub fn evaluate<S, A, E, M, T, F>(
        game_name: &str,
        run_name: &str,
        model_factory: F,
        game_engine: &E,
        num_games_to_play: usize,
        parallelism: usize,
        options: &SelfEvaluateOptions
    ) -> Result<(), &'static str> 
    where
        S: GameState,
        A: Clone + Eq + DeserializeOwned + Serialize + Debug + Unpin + Send,
        E: GameEngine<State=S,Action=A> + Sync,
        M: Model<State=S,Action=A,Analyzer=T>,
        T: GameAnalyzer<Action=A,State=S> + Send,
        F: ModelFactory<M=M>
    {
        // let run_directory = Self::get_run_directory(&game_name, &run_name);

        let model_1_info = &ModelInfo::new(game_name.to_owned(), run_name.to_owned(), 1);
        let model_2_info = &ModelInfo::new(game_name.to_owned(), run_name.to_owned(), 2);

        let model_1 = model_factory.get(model_1_info);
        let model_2 = model_factory.get(model_2_info);

        let starting_time = Instant::now();

        let starting_run_time = Instant::now();
        let (game_results_tx, game_results_rx) = std::sync::mpsc::channel();

        crossbeam::scope(move |s| {
            for thread_num in 0..parallelism {
                let game_results_tx = game_results_tx.clone();

                // @TODO: UPDATE
                let analyzer_1 = model_1.get_game_state_analyzer();
                let analyzer_2 = model_2.get_game_state_analyzer();

                s.spawn(move |_| {
                    println!("Starting Thread: {}", thread_num);

                    let f = Self::play_games(
                        num_games_to_play,
                        game_results_tx,
                        game_engine,
                        (model_1_info, &analyzer_1),
                        (model_2_info, &analyzer_2),
                        options
                    ).map(|_| ());

                    tokio_current_thread::block_on_all(f);
                });
            }

            s.spawn(move |_| -> Result<(), &'static str> {
                let mut num_of_games_played: usize = 0;

                while let Ok(match_result) = game_results_rx.recv() {
                    num_of_games_played += 1;

                    println!("{:?}", match_result);

                    println!(
                        "Time Elapsed: {:.2}h, Number of Games Played: {}, GPM: {:.2}",
                        starting_time.elapsed().as_secs() as f64 / (60 * 60) as f64,
                        num_of_games_played,
                        num_of_games_played as f64 / starting_run_time.elapsed().as_secs() as f64 * 60 as f64
                    );
                }

                Ok(())
            });
        }).map_err(|_| "Failed to spawn self play threads")?;

        Ok(())
    }

    async fn play_games<S, A, E, T>(
        num_games_to_play: usize,
        results_channel: mpsc::Sender<MatchResult<A>>,
        game_engine: &E,
        model_1: (&ModelInfo, &T),
        model_2: (&ModelInfo, &T),
        options: &SelfEvaluateOptions
    ) -> Result<(), &'static str>
    where
        S: GameState,
        A: Clone + Eq + DeserializeOwned + Serialize + Debug + Unpin + Send,
        E: GameEngine<State=S,Action=A> + Sync,
        T: GameAnalyzer<Action=A,State=S> + Send
    {
        let mut match_result_stream = FuturesUnordered::new();

        for i in 0..num_games_to_play {
            let (p1, p2) = if i % 2 == 0 {
                (model_1, model_2)
            } else {
                (model_2, model_1)
            };

            match_result_stream.push(
                Self::play_game(game_engine, p1, p2, options)
            );
        }

        while let Some(match_result) = match_result_stream.next().await {
            let match_result = match_result.unwrap();

            results_channel.send(match_result).expect("Failed to send match result");
        }

        Ok(())
    }

    #[allow(non_snake_case)]
    async fn play_game<S, A, E, T>(
        game_engine: &E,
        model_1: (&ModelInfo, &T),
        model_2: (&ModelInfo, &T),
        options: &SelfEvaluateOptions
    ) -> Result<MatchResult<A>, &'static str>
    where
        S: GameState,
        A: Clone + Eq + DeserializeOwned + Serialize + Debug + Unpin + Send,
        E: GameEngine<State=S,Action=A> + Sync,
        T: GameAnalyzer<Action=A,State=S> + Send
    {
        let uuid = Uuid::new_v4();
        let cpuct_base = options.cpuct_base;
        let cpuct_init = options.cpuct_init;
        let temperature_max_actions = options.temperature_max_actions;
        let temperature = options.temperature;
        let temperature_post_max_actions = options.temperature_post_max_actions;
        let visits = options.visits;
        let mut p1_last_to_move = false;

        let mut mcts_1 = MCTS::new(
            S::initial(),
            List::new(),
            game_engine,
            model_1.1,
            MCTSOptions::<S,A,_,_,_>::new(
                None,
                |_,_,_,Nsb| ((Nsb as f64 + cpuct_base + 1.0) / cpuct_base).ln() + cpuct_init,
                |_,actions| if actions.len() < temperature_max_actions { temperature } else { temperature_post_max_actions },
                rng::create_rng_from_uuid(uuid),
            )
        );

        let mut mcts_2 = MCTS::new(
            S::initial(),
            List::new(),
            game_engine,
            model_2.1,
            MCTSOptions::<S,A,_,_,_>::new(
                None,
                |_,_,_,Nsb| ((Nsb as f64 + cpuct_base + 1.0) / cpuct_base).ln() + cpuct_init,
                |_,actions| if actions.len() < temperature_max_actions { temperature } else { temperature_post_max_actions },
                rng::create_rng_from_uuid(uuid),
            )
        );

        let mut actions: Vec<A> = Vec::new();
        let mut state: S = S::initial();

        while game_engine.is_terminal_state(&state) == None {
            let search_result = if !p1_last_to_move {
                mcts_1.search(visits).await?
            } else {
                mcts_2.search(visits).await?
            };

            let action = search_result.0;

            mcts_1.advance_to_action(action.to_owned()).await?;
            mcts_2.advance_to_action(action.to_owned()).await?;

            state = game_engine.take_action(&state, &action);

            actions.push(action);

            p1_last_to_move = !p1_last_to_move;
        };

        let final_score = game_engine.is_terminal_state(&state).ok_or("Expected a terminal state")?;
        let score = if p1_last_to_move { final_score * -1.0 } else { final_score };

        Ok(MatchResult {
            guid: uuid.to_string(),
            p1_model_num: model_1.0.get_run_num(),
            p2_model_num: model_2.0.get_run_num(),
            actions,
            score
        })
    }
}
