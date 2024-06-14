use anyhow::{anyhow, Result};
use rand::Rng;
use std::fmt::Debug;

use engine::engine::GameEngine;
use engine::game_state::GameState;
use mcts::{BackpropagationStrategy, DirichletOptions, SelectionStrategy, TemperatureConstant, MCTS};
use model::GameAnalyzer;

use super::{SelfPlayMetrics, SelfPlayOptions};

#[allow(non_snake_case)]
pub async fn play_self_one<S, A, E, M, B, Sel, P, PV>(
    game_engine: &E,
    analyzer: &M,
    backpropagation_strategy: &B,
    selection_strategy: &Sel,
    options: &SelfPlayOptions,
) -> Result<(SelfPlayMetrics<A, P, PV>, S)>
where
    S: GameState,
    A: Clone + Eq + Debug,
    E: GameEngine<State = S, Action = A, Terminal = P>,
    M: GameAnalyzer<State = S, Action = A, Predictions = P>,
    B: BackpropagationStrategy<State = S, Action = A, Predictions = P, PropagatedValues = PV>,
    Sel: SelectionStrategy<State = S, Action = A, Predictions = P, PropagatedValues = PV>,
    P: Clone,
    PV: Default + Ord + Clone
{
    let mut game_state: S = S::initial();
    let play_options = &options.play_options;
    let dirichlet_options = Some(DirichletOptions {
        epsilon: options.epsilon,
    });

    // @TODO: Verify options
    // let cpuct = DynamicCPUCT::new(
    //     play_options.cpuct_base,
    //     play_options.cpuct_init,
    //     1.0,
    //     play_options.cpuct_root_scaling,
    // );

    // MCTSOptions::new(
    //     Some(DirichletOptions {
    //         epsilon: options.epsilon,
    //     }),
    //     play_options.fpu,
    //     play_options.fpu_root,
    //     play_options.temperature_visit_offset,
    //     play_options.moves_left_threshold,
    //     play_options.moves_left_scale,
    //     play_options.moves_left_factor,
    //     play_options.parallelism,
    // ),

    let temp = TemperatureConstant::new(play_options.temperature);

    let mut mcts = MCTS::with_capacity(
        game_state.clone(),
        game_engine,
        analyzer,
        backpropagation_strategy,
        selection_strategy,
        options.visits,
        temp,
        play_options.parallelism
    );

    let mut analysis = Vec::new();
    let mut rng = rand::thread_rng();

    while game_engine.terminal_state(&game_state).is_none() {
        let action = if rng.gen::<f32>() <= options.full_visits_probability {
            mcts.apply_noise_at_root(dirichlet_options.as_ref()).await;
            mcts.search_visits(options.visits).await?;
            mcts.select_action_with_temp(play_options.temperature_visit_offset)?
        } else {
            mcts.search_visits(options.fast_visits).await?;
            mcts.select_action()?
        };

        let metrics = mcts.get_root_node_metrics()?;

        mcts.advance_to_action(action.to_owned()).await?;
        game_state = game_engine.take_action(&game_state, &action);
        analysis.push((action, metrics));
    }

    let score = game_engine
        .terminal_state(&game_state)
        .ok_or_else(|| anyhow!("Expected a terminal state"))?;

    Ok((SelfPlayMetrics::<A, P, PV>::new(analysis, score), game_state))
}
