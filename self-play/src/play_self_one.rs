use anyhow::{anyhow, Result};
use rand::Rng;
use std::fmt::Debug;

use engine::engine::GameEngine;
use engine::game_state::GameState;
use engine::value::Value;
use mcts::mcts::{DirichletOptions, MCTSOptions, MCTS};
use model::GameAnalyzer;

use super::{SelfPlayMetrics, SelfPlayOptions};

#[allow(non_snake_case)]
pub async fn play_self_one<S, A, E, M, V>(
    game_engine: &E,
    analyzer: &M,
    options: &SelfPlayOptions,
) -> Result<(SelfPlayMetrics<A, V>, S)>
where
    S: GameState,
    A: Clone + Eq + Debug,
    V: Value,
    E: GameEngine<State = S, Action = A, Value = V>,
    M: GameAnalyzer<State = S, Action = A, Value = V>,
{
    let mut game_state: S = S::initial();
    let play_options = &options.play_options;
    let cpuct_base = play_options.cpuct_base;
    let cpuct_init = play_options.cpuct_init;
    let cpuct_root_scaling = play_options.cpuct_root_scaling;

    let mut mcts = MCTS::with_capacity(
        game_state.clone(),
        game_engine,
        analyzer,
        MCTSOptions::<S, _, _>::new(
            Some(DirichletOptions {
                epsilon: options.epsilon,
            }),
            play_options.fpu,
            play_options.fpu_root,
            |_, Nsb, is_root| {
                (((Nsb as f32 + cpuct_base + 1.0) / cpuct_base).ln() + cpuct_init)
                    * if is_root { cpuct_root_scaling } else { 1.0 }
            },
            |game_state| {
                let move_number = game_engine.get_move_number(game_state);
                if move_number < play_options.temperature_max_moves {
                    play_options.temperature
                } else {
                    play_options.temperature_post_max_moves
                }
            },
            play_options.temperature_visit_offset,
            play_options.moves_left_threshold,
            play_options.moves_left_scale,
            play_options.moves_left_factor,
            play_options.parallelism,
        ),
        options.visits,
    );

    let mut analysis = Vec::new();
    let mut rng = rand::thread_rng();

    while game_engine.is_terminal_state(&game_state).is_none() {
        let action = if rng.gen::<f32>() <= options.full_visits_probability {
            mcts.apply_noise_at_root().await;
            mcts.search_visits(options.visits).await?;
            mcts.select_action()?
        } else {
            mcts.search_visits(options.fast_visits).await?;
            mcts.select_action_no_temp()?
        };

        let metrics = mcts.get_root_node_metrics()?;

        mcts.advance_to_action(action.to_owned()).await?;
        game_state = game_engine.take_action(&game_state, &action);
        analysis.push((action, metrics));
    }

    let score = game_engine
        .is_terminal_state(&game_state)
        .ok_or_else(|| anyhow!("Expected a terminal state"))?;

    Ok((SelfPlayMetrics::<A, V>::new(analysis, score), game_state))
}
