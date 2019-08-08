use std::fmt::Debug;
use std::time::Instant;
use std::path::{PathBuf};
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
use model::model::{Model, ModelFactory};

pub struct SelfEvaluate<'a, S, A, E, M, T>
where
    S: GameState,
    A: Clone + Eq + Serialize + Unpin,
    E: 'a + GameEngine<State=S,Action=A>,
    M: 'a + Model<Action=A,State=S,Analyzer=T>,
    T: GameAnalyzer<Action=A,State=S> + Send
{
    run_directory: PathBuf,
    latest_model: M,
    game_engine: &'a E
}

impl<'a, S, A, E, M, T> SelfEvaluate<'a, S, A, E, M, T>
where
    S: GameState,
    A: Clone + Eq + DeserializeOwned + Serialize + Debug + Unpin + Send,
    E: 'a + GameEngine<State=S,Action=A> + Sync,
    M: 'a + Model<State=S,Action=A,Analyzer=T>,
    T: GameAnalyzer<Action=A,State=S> + Send
{
    pub fn evaluate<F>(
        game_name: String,
        run_name: String,
        model_factory: F,
        game_engine: &'a E
    ) -> Result<Self, &'static str> 
    where
        F: ModelFactory<M=M>
    {
        let run_directory = Self::get_run_directory(&game_name, &run_name);
        let a_name = Self::get_model_name(&game_name, &run_name, 1);
        let latest_model = model_factory.get_latest(&a_name);

        Ok(Self {
            run_directory,
            latest_model,
            game_engine
        })
    }

    pub fn learn(&mut self) -> Result<(), &'static str> {
        let run_directory = &self.run_directory;
        let starting_time = Instant::now();

        let starting_run_time = Instant::now();
        let latest_model = &self.latest_model;
        let model_name = latest_model.get_name();
        let (game_results_tx, game_results_rx) = std::sync::mpsc::channel();

        let mut num_games_to_play = 1000;

        let game_engine = self.game_engine;

        crossbeam::scope(move |s| {
            for thread_num in 0..3 {
                let game_results_tx = game_results_tx.clone();
                let analyzer = latest_model.get_game_state_analyzer();

                s.spawn(move |_| {
                    println!("Starting Thread: {}", thread_num);

                    let f = Self::play_games(
                        num_games_to_play,
                        game_results_tx,
                        game_engine,
                        analyzer,
                        analyzer
                    ).map(|_| ());

                    tokio_current_thread::block_on_all(f);
                });
            }

            s.spawn(move |_| -> Result<(), &'static str> {
                let mut num_of_games_played: usize = 0;

                while let Ok(self_play_metric) = game_results_rx.recv() {
                    num_of_games_played += 1;
                    num_games_to_play -= 1;

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

    async fn play_games(
        num_games_to_play: usize,
        results_channel: mpsc::Sender<SelfPlayMetrics<A>>,
        game_engine: &E,
        analyzer_1: T,
        analyzer_2: T
    ) -> Result<(), &'static str> {
        let mut match_result_stream = FuturesUnordered::new();

        for _ in 0..num_games_to_play {
            match_result_stream.push(
                Self::play_game(game_engine, &analyzer_1, &analyzer_2)
            );
        }

        while let Some(match_result) = match_result_stream.next().await {
            let match_result = match_result.unwrap();

            results_channel.send(match_result).expect("Failed to send match result");
        }

        Ok(())
    }

    async fn play_game(
        game_engine: &E,
        analyzer_1: &T,
        analyzer_2: &T
    ) -> Result<f64, &'static str> {
        let uuid = Uuid::new_v4();
        let seedable_rng = rng::create_rng_from_uuid(uuid);
        let cpuct_base: f64 = 19_652.0;
        let cpuct_init: f64 = 1.25;
        let temperature_max_actions: usize = 16;
        let temperature: f64 = 0.45;
        let temperature_post_max_actions: f64 = 0.0;
        let visits: usize = 800;
        let p1_last_to_move = false;

        let mut mcts_1 = MCTS::new(
            S::initial(),
            List::new(),
            game_engine,
            analyzer_1,
            MCTSOptions::<S,A,_,_,_>::new(
                None,
                |_,_,_,Nsb| ((Nsb as f64 + cpuct_base + 1.0) / cpuct_base).ln() + cpuct_init,
                |_,actions| if actions.len() < temperature_max_actions { temperature } else { temperature_post_max_actions },
                seedable_rng,
            )
        );

        let mut mcts_2 = MCTS::new(
            S::initial(),
            List::new(),
            game_engine,
            analyzer_2,
            MCTSOptions::<S,A,_,_,_>::new(
                None,
                |_,_,_,Nsb| ((Nsb as f64 + cpuct_base + 1.0) / cpuct_base).ln() + cpuct_init,
                |_,actions| if actions.len() < temperature_max_actions { temperature } else { temperature_post_max_actions },
                seedable_rng,
            )
        );

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

            p1_last_to_move = !p1_last_to_move;
        };

        let final_score = game_engine.is_terminal_state(&state).ok_or("Expected a terminal state")?;
        let final_score_p1 = if p1_last_to_move { final_score * -1.0 } else { final_score };

        Ok(final_score_p1)
    }

    fn get_model_name(game_name: &str, run_name: &str, model_number: usize) -> String {
        format!("{}_{}_{:0>5}", game_name, run_name, model_number)
    }

    fn get_run_directory(game_name: &str, run_name: &str) -> PathBuf {
        PathBuf::from(format!("./{}_runs/{}", game_name, run_name))
    }
}
