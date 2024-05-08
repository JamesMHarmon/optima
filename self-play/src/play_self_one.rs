use anyhow::{anyhow, Result};
use rand::Rng;
use std::fmt::Debug;

use engine::engine::GameEngine;
use engine::game_state::GameState;
use engine::value::Value;
use mcts::{DirichletOptions, DynamicCPUCT, MCTSOptions, TemperatureConstant, MCTS};
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

    let cpuct = DynamicCPUCT::new(
        play_options.cpuct_base,
        play_options.cpuct_init,
        1.0,
        play_options.cpuct_root_scaling,
    );

    let temp = TemperatureConstant::new(play_options.temperature);

    let mut mcts = MCTS::with_capacity(
        game_state.clone(),
        game_engine,
        analyzer,
        MCTSOptions::new(
            Some(DirichletOptions {
                epsilon: options.epsilon,
            }),
            play_options.fpu,
            play_options.fpu_root,
            play_options.temperature_visit_offset,
            play_options.moves_left_threshold,
            play_options.moves_left_scale,
            play_options.moves_left_factor,
            play_options.parallelism,
        ),
        options.visits,
        cpuct,
        temp,
    );

    let mut analysis = Vec::new();
    let mut rng = rand::thread_rng();

    while game_engine.terminal_state(&game_state).is_none() {
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
        .terminal_state(&game_state)
        .ok_or_else(|| anyhow!("Expected a terminal state"))?;

    Ok((SelfPlayMetrics::<A, V>::new(analysis, score), game_state))
}
