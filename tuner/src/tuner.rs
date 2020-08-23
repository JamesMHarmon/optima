use model::model::ModelFactory;
use std::path::PathBuf;
use std::fmt::Debug;
use std::time::Instant;
use std::sync::mpsc;
use serde::{Deserialize,Serialize};
use serde::de::{DeserializeOwned};
use futures::stream::{FuturesUnordered,StreamExt};
use anyhow::{Result,anyhow};
use itertools::Itertools;
use log::info;

use mcts::mcts::{MCTS,MCTSOptions};
use model::analytics::GameAnalyzer;
use engine::engine::GameEngine;
use engine::game_state::GameState;
use engine::value::Value;
use model::model::{Model};
use model::model_info::{ModelInfo};

use super::tuner_persistance::TunerPersistance;
use super::constants::TUNER_PARALLELISM;

#[derive(Serialize, Deserialize, Debug)]
pub struct TunerOptions<'a> {
    pub name: &'a str,
    pub num_players: usize,
    pub num_rounds: usize,
    pub batch_size: usize,
}

#[derive(Debug,Serialize)]
pub struct PlayerOptions {
    pub name: String,
    pub parallelism: usize,
    pub temperature: f32,
    pub temperature_max_actions: usize,
    pub temperature_post_max_actions: f32,
    pub visits: usize,
    pub fpu: f32,
    pub fpu_root: f32,
    pub logit_q: bool,
    pub cpuct_base: f32,
    pub cpuct_init: f32,
    pub cpuct_root_scaling: f32,
    pub moves_left_threshold:  f32,
    pub moves_left_scale:  f32,
    pub moves_left_factor: f32,
    pub model_info: ModelInfo,
}

pub struct Player<'a,T> {
    id: usize,
    options: &'a PlayerOptions,
    analyzer: T
}

pub struct Tuner {}

#[derive(Debug,Serialize)]
pub struct GameResult<A> {
    actions: Vec<A>,
    scores: Vec<(usize, f32)>
}

#[derive(Debug,Serialize)]
pub struct PlayerScore<'a> {
    pub player: &'a PlayerOptions,
    pub score: f32
}

impl Tuner
{
    pub fn tuner<S,A,E,M,F,T>(
        players: &[PlayerOptions],
        model_factory: &F,
        game_engine: &E,
        options: &TunerOptions<'static>
    ) -> Result<()> 
    where
        S: GameState,
        A: Clone + Eq + DeserializeOwned + Serialize + Debug + Unpin + Send,
        E: GameEngine<State=S,Action=A,Value=M::Value> + Sync,
        M: Model<State=S,Action=A,Analyzer=T>,
        F: ModelFactory<M=M>,
        T: GameAnalyzer<Action=A,State=S,Value=M::Value> + Send
    {
        let starting_time = Instant::now();

        let starting_run_time = Instant::now();
        let (game_results_tx, game_results_rx) = std::sync::mpsc::channel();

        let batch_size = options.batch_size;
        let players = players.iter().enumerate().collect::<Vec<_>>();

        crossbeam::scope(move |s| {
            let mut model_map = std::collections::HashMap::new();

            let games_to_play = generate_games_to_play(&players, options.num_players, options.num_rounds);
            let games_to_play = games_to_play.iter()
                .map(|players| players.iter().map(|(p_id, p)| {
                    let model_info = &p.model_info;
                    let run_name = model_info.get_run_name();
                    let model_num = model_info.get_model_num();
                    let key = format!("{}_{}", run_name, model_num);

                    let model = model_map.entry(key).or_insert_with(|| model_factory.get(&model_info));

                    Player {
                        id: *p_id,
                        options: *p,
                        analyzer: model.get_game_state_analyzer()
                    }
                }).collect::<Vec<_>>());

            let num_games_to_play = games_to_play.len();
            let num_games_per_thread = num_games_to_play / TUNER_PARALLELISM;
            let num_games_per_thread = num_games_per_thread.max(num_games_to_play);

            let games_to_play_chunks = games_to_play.into_iter().chunks(num_games_per_thread);

            for (thread_num, games_to_play) in games_to_play_chunks.into_iter().map(|c| c.collect::<Vec<_>>()).enumerate() {
                let game_results_tx = game_results_tx.clone();
                let num_games_to_play_this_thread = games_to_play.len();

                s.spawn(move |_| {
                    info!("Starting Thread: {}, Games: {}", thread_num, num_games_to_play_this_thread);

                    let f = Self::play_games(
                        games_to_play,
                        batch_size,
                        game_results_tx,
                        game_engine
                    );

                    common::runtime::block_on(f).unwrap();
                });
            }

            s.spawn(move |_| -> Result<()> {
                let mut num_of_games_played: usize = 0;
                let mut player_scores: Vec<_> = players.iter().map(|(id, m)| (*id, *m, 0.0)).collect();

                let model_info = &players[0].1.model_info;
                let game_name = model_info.get_game_name();
                let run_name = model_info.get_run_name();

                let mut presistance = TunerPersistance::new(
                    &get_run_directory(game_name, run_name),
                    options.name
                )?;

                while let Ok(game_result) = game_results_rx.recv() {
                    num_of_games_played += 1;

                    for (player_id, score) in game_result.scores.iter() {
                        player_scores.iter_mut().find(|(id, _, _)| *id == *player_id).unwrap().2 += score;
                    }

                    info!("{:?}", game_result);

                    info!(
                        "Time Elapsed: {:.2}h, Number of Games Played: {}, GPM: {:.2}",
                        starting_time.elapsed().as_secs() as f32 / (60 * 60) as f32,
                        num_of_games_played,
                        num_of_games_played as f32 / starting_run_time.elapsed().as_secs() as f32 * 60_f32
                    );

                    presistance.write_game(&game_result)?;
                }

                let player_scores = player_scores.iter().map(|(_, player, score)| PlayerScore { player, score: *score }).collect::<Vec<_>>();
                presistance.write_player_scores(&player_scores)?;

                Ok(())
            });
        }).unwrap();

        Ok(())
    }

    async fn play_games<'a, S, A, E, T>(
        games_to_play: Vec<Vec<Player<'a, T>>>,
        batch_size: usize,
        results_channel: mpsc::Sender<GameResult<A>>,
        game_engine: &E
    ) -> Result<()>
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
                let game_to_play = Self::play_game(game_engine, players);
                game_result_stream.push(game_to_play);
            }
        }

        while let Some(game_result) = game_result_stream.next().await {
            let game_result = game_result?;

            results_channel.send(game_result).map_err(|_| anyhow!("Failed to send game_result"))?;

            if let Some(players) = games_to_play.pop() {
                let game_to_play = Self::play_game(game_engine, players);
                game_result_stream.push(game_to_play);
            }
        }

        Ok(())
    }

    #[allow(non_snake_case)]
    async fn play_game<'a, S, A, E, T>(
        game_engine: &E,
        players: Vec<Player<'a, T>>
    ) -> Result<GameResult<A>>
    where
        S: GameState,
        A: Clone + Eq + DeserializeOwned + Serialize + Debug + Unpin + Send,
        E: GameEngine<State=S,Action=A,Value=T::Value> + Sync,
        T: GameAnalyzer<Action=A,State=S> + Send
    {

        let mut mctss: Vec<_> = players.iter().map(|Player { analyzer, options, .. }| {
            let cpuct_base = options.cpuct_base;
            let cpuct_init = options.cpuct_init;
            let cpuct_root_scaling = options.cpuct_root_scaling;
            let temperature = options.temperature;
            let temperature_max_actions = options.temperature_max_actions;
            let temperature_post_max_actions = options.temperature_post_max_actions;

            (MCTS::with_capacity(
                S::initial(),
                0,
                game_engine,
                analyzer,
                MCTSOptions::<S,_,_>::new(
                    None,
                    options.fpu,
                    options.fpu_root,
                    options.logit_q,
                    move |_,_,Nsb,is_root| (((Nsb as f32 + cpuct_base + 1.0) / cpuct_base).ln() + cpuct_init) * if is_root { cpuct_root_scaling } else { 1.0 },
                    move |_,num_actions| if num_actions < temperature_max_actions { temperature } else { temperature_post_max_actions },
                    0.0,
                    options.moves_left_threshold,
                    options.moves_left_scale,
                    options.moves_left_factor,
                    options.parallelism
                ),
                options.visits
            ), options.visits)
        }).collect();

        let mut actions: Vec<A> = Vec::new();
        let mut state: S = S::initial();

        while game_engine.is_terminal_state(&state).is_none() {
            let player_to_move = game_engine.get_player_to_move(&state);
            let (player_to_move_mcts, player_to_move_visits) = &mut mctss[player_to_move - 1];
            player_to_move_mcts.search_visits(*player_to_move_visits).await?;
            let action = player_to_move_mcts.select_action()?;
            
            for (mcts, _) in &mut mctss {
                mcts.advance_to_action(action.to_owned()).await?;
            }

            state = game_engine.take_action(&state, &action);

            actions.push(action);
        };

        let final_score = game_engine.is_terminal_state(&state).ok_or_else(|| anyhow!("Expected a terminal state"))?;

        let scores: Vec<_> = players.iter().enumerate().map(|(i, Player { id, .. })| (
            *id,
            final_score.get_value_for_player(i + 1))
        ).collect();

        Ok(GameResult {
            actions,
            scores
        })
    }
}

fn generate_games_to_play<'a>(
    players: &[(usize, &'a PlayerOptions)],
    num_players: usize,
    num_rounds: usize
) -> Vec<Vec<(usize, &'a PlayerOptions)>>
{
    let games_to_play: Vec<Vec<_>> = std::iter::repeat_with(||
            players.iter().permutations(num_players)
        )
        .take(num_rounds)
        .flatten()
        .map(|players| players.iter().map(|(id, options)| (*id, *options)).collect())
        .collect();

    games_to_play
}

fn get_run_directory(game_name: &str, run_name: &str) -> PathBuf {
    PathBuf::from(format!("./{}_runs/{}", game_name, run_name))
}
