use std::path::PathBuf;
use std::fmt::Debug;
use std::time::Instant;
use std::sync::mpsc;
use serde::{Deserialize,Serialize};
use serde::de::{DeserializeOwned};
use futures::stream::{FuturesUnordered,StreamExt};
use failure::{Error,format_err};
use tokio_executor::current_thread;
use itertools::Itertools;

use mcts::mcts::{MCTS,MCTSOptions};
use model::analytics::GameAnalyzer;
use engine::engine::GameEngine;
use engine::game_state::GameState;
use engine::value::Value;
use model::model::{Model};
use model::model_info::{ModelInfo};

use super::tournament_persistance::TournamentPersistance;
use super::constants::TOURNAMENT_PARALLELISM;

#[derive(Serialize, Deserialize, Debug)]
pub struct TournamentOptions {
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
    pub cpuct_root_scaling: f32,
    pub num_players: usize
}

pub struct Tournament {}

#[derive(Debug,Serialize)]
pub struct GameResult<A> {
    actions: Vec<A>,
    scores: Vec<(ModelInfo, f32)>
}

#[derive(Debug,Serialize)]
pub struct ModelScore {
    model: ModelInfo,
    score: f32
}

impl Tournament
{
    pub fn tournament<S,A,E,M,T>(
        models: &[M],
        game_engine: &E,
        options: &TournamentOptions
    ) -> Result<(), Error> 
    where
        S: GameState,
        A: Clone + Eq + DeserializeOwned + Serialize + Debug + Unpin + Send,
        E: GameEngine<State=S,Action=A,Value=M::Value> + Sync,
        M: Model<State=S,Action=A,Analyzer=T>,
        T: GameAnalyzer<Action=A,State=S,Value=M::Value> + Send
    {
        let starting_time = Instant::now();

        let starting_run_time = Instant::now();
        let (game_results_tx, game_results_rx) = std::sync::mpsc::channel();

        let batch_size = options.batch_size;

        crossbeam::scope(move |s| {
            let games_to_play = generate_games_to_play(models, options.num_players).iter()
                .map(|models| models.iter().map(|m| (m.get_model_info(), m.get_game_state_analyzer())).collect::<Vec<_>>())
                .collect::<Vec<_>>();

            let num_games_to_play = games_to_play.len();
            let num_games_per_thread = num_games_to_play / TOURNAMENT_PARALLELISM;
            let games_to_play_chunks = games_to_play.into_iter()
                .chunks(num_games_per_thread)
                .into_iter().map(|c| c.collect::<Vec<_>>())
                .collect::<Vec<_>>();

            for (thread_num, games_to_play) in games_to_play_chunks.into_iter().enumerate() {
                let game_results_tx = game_results_tx.clone();
                let num_games_to_play_this_thread = games_to_play.len();

                s.spawn(move |_| {
                    println!("Starting Thread: {}, Games: {}", thread_num, num_games_to_play_this_thread);

                    let f = Self::play_games(
                        games_to_play,
                        batch_size,
                        game_results_tx,
                        game_engine,
                        options
                    );

                    current_thread::block_on_all(f).unwrap();
                });
            }

            let model_info: Vec<_> = models.iter().map(|m| m.get_model_info().to_owned()).collect();
            s.spawn(move |_| -> Result<(), Error> {
                let mut num_of_games_played: usize = 0;
                let mut model_scores: Vec<_> = model_info.iter().map(|m| (m.to_owned(), 0.0)).collect();

                let game_name = model_info[0].get_game_name();
                let run_name = model_info[0].get_run_name();

                let mut presistance = TournamentPersistance::new(
                    &get_run_directory(game_name, run_name),
                    &model_info
                )?;

                while let Ok(game_result) = game_results_rx.recv() {
                    num_of_games_played += 1;

                    for (model_info, score) in game_result.scores.iter() {
                        model_scores.iter_mut().find(|(m, _s)| *m == *model_info).unwrap().1 += score;
                    }

                    println!("{:?}", game_result);

                    println!(
                        "Time Elapsed: {:.2}h, Number of Games Played: {}, GPM: {:.2}",
                        starting_time.elapsed().as_secs() as f32 / (60 * 60) as f32,
                        num_of_games_played,
                        num_of_games_played as f32 / starting_run_time.elapsed().as_secs() as f32 * 60 as f32
                    );

                    presistance.write_game(&game_result)?;
                }

                presistance.write_model_scores(&model_scores)?;

                Ok(())
            });
        }).unwrap();

        Ok(())
    }

    async fn play_games<S, A, E, T>(
        games_to_play: Vec<Vec<(&ModelInfo, T)>>,
        batch_size: usize,
        results_channel: mpsc::Sender<GameResult<A>>,
        game_engine: &E,
        options: &TournamentOptions
    ) -> Result<(), Error>
    where
        S: GameState,
        A: Clone + Eq + DeserializeOwned + Serialize + Debug + Unpin + Send,
        E: GameEngine<State=S,Action=A,Value=T::Value> + Sync,
        T: GameAnalyzer<Action=A,State=S> + Send
    {
        let mut games_to_play = games_to_play;
        let mut game_result_stream = FuturesUnordered::new();

        for _ in 0..batch_size {
            if let Some(players) = games_to_play.pop() {
                let game_to_play = Self::play_game(game_engine, players, options);
                game_result_stream.push(game_to_play);
            }
        }

        while let Some(game_result) = game_result_stream.next().await {
            let game_result = game_result?;

            results_channel.send(game_result).map_err(|_| format_err!("Failed to send game_result"))?;

            if let Some(players) = games_to_play.pop() {
                let game_to_play = Self::play_game(game_engine, players, options);
                game_result_stream.push(game_to_play);
            }
        }

        Ok(())
    }

    #[allow(non_snake_case)]
    async fn play_game<S, A, E, T>(
        game_engine: &E,
        players: Vec<(&ModelInfo, T)>,
        options: &TournamentOptions
    ) -> Result<GameResult<A>, Error>
    where
        S: GameState,
        A: Clone + Eq + DeserializeOwned + Serialize + Debug + Unpin + Send,
        E: GameEngine<State=S,Action=A,Value=T::Value> + Sync,
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

        let mut mctss: Vec<_> = players.iter().map(|(_, analyzer)| MCTS::with_capacity(
            S::initial(),
            0,
            game_engine,
            analyzer,
            MCTSOptions::<S,_,_>::new(
                None,
                fpu,
                fpu_root,
                |_,_,Nsb,is_root| (((Nsb as f32 + cpuct_base + 1.0) / cpuct_base).ln() + cpuct_init) * if is_root { cpuct_root_scaling } else { 1.0 },
                |_,num_actions| if num_actions < temperature_max_actions { temperature } else { temperature_post_max_actions },
                0.0,
                options.parallelism
            ),
            visits
        )).collect();

        let mut actions: Vec<A> = Vec::new();
        let mut state: S = S::initial();

        while game_engine.is_terminal_state(&state).is_none() {
            let player_to_move = game_engine.get_player_to_move(&state);
            let player_to_move_mcts = &mut mctss[player_to_move - 1];
            player_to_move_mcts.search_visits(visits).await?;
            let action = player_to_move_mcts.select_action().await?;
            
            for mcts in &mut mctss {
                mcts.advance_to_action(action.to_owned()).await?;
            }

            state = game_engine.take_action(&state, &action);

            actions.push(action);
        };

        let final_score = game_engine.is_terminal_state(&state).ok_or(format_err!("Expected a terminal state"))?;

        let scores: Vec<_> = players.iter().enumerate().map(|(i, (m, _))| (
            (**m).to_owned(),
            final_score.get_value_for_player(i + 1))
        ).collect();

        Ok(GameResult {
            actions,
            scores
        })
    }
}

fn generate_games_to_play<S,A,M,T>(
    models: &[M],
    num_players: usize
) -> Vec<Vec<&M>>
where
    S: engine::game_state::GameState,
    M: Model<State=S,Action=A,Analyzer=T>,
    T: GameAnalyzer<Action=A,State=S,Value=M::Value> + Send,
{
    let games_to_play: Vec<Vec<(&M)>> = models.iter()
        .permutations(num_players)
        .collect();

    games_to_play
}

fn get_run_directory(game_name: &str, run_name: &str) -> PathBuf {
    PathBuf::from(format!("./{}_runs/{}", game_name, run_name))
}
