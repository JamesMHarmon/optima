use anyhow::{Result, anyhow};
use rand::Rng;
use std::fmt::Debug;

use common::{GameLength, PlayerToMove, TranspositionHash};
use engine::{GameEngine, GameState};
use mcts::SnapshotToPropagated;
use mcts::{DirichletOptions, NoTemp, PuctMCTS, TemperatureConstant};
use model::GameAnalyzer;
use puct::{RollupStats, SelectionPolicy, ValueModel};

use super::{SelfPlayMetrics, SelfPlayOptions};

type SnapshotOf<VM> = <<VM as ValueModel>::Rollup as RollupStats>::Snapshot;
type PVOf<VM> = <SnapshotOf<VM> as SnapshotToPropagated>::PropagatedValues;

#[allow(non_snake_case)]
pub async fn play_self_one<S, A, E, M, P, VM, Sel>(
    game_engine: &E,
    analyzer: &M,
    value_model: &VM,
    selection: &Sel,
    options: &SelfPlayOptions,
) -> Result<(SelfPlayMetrics<A, P, PVOf<VM>>, S)>
where
    S: GameState + Clone + TranspositionHash + PlayerToMove,
    A: Clone + Eq + Debug,
    E: GameEngine<State = S, Action = A, Terminal = P>,
    M: GameAnalyzer<State = S, Action = A, Predictions = P>,
    P: Clone + engine::Value + GameLength,
    VM: ValueModel<State = S, Predictions = P, Terminal = P>,
    Sel: SelectionPolicy<SnapshotOf<VM>, State = S>,
    SnapshotOf<VM>: Clone + SnapshotToPropagated,
{
    let mut game_state: S = S::initial();
    let play_options = &options.play_options;
    let dirichlet_options = Some(DirichletOptions {
        epsilon: options.epsilon,
    });

    let temp = TemperatureConstant::new(
        play_options.temperature,
        play_options.temperature_visit_offset,
    );
    let no_temp = NoTemp::new();

    let mut mcts = PuctMCTS::new(
        game_state.clone(),
        game_engine,
        analyzer,
        value_model,
        selection,
    );

    let mut analysis = Vec::new();
    let mut rng = rand::rng();

    while game_engine.terminal_state(&game_state).is_none() {
        let action = if rng.random::<f32>() <= options.full_visits_probability {
            mcts.apply_noise_at_root(dirichlet_options.as_ref()).await;
            mcts.search_visits(options.visits).await?;
            mcts.select_action(&temp)?
        } else {
            mcts.search_visits(options.fast_visits).await?;
            mcts.select_action(&no_temp)?
        };

        let metrics = mcts.get_root_node_metrics()?;

        mcts.advance_to_action(action.to_owned()).await?;
        game_state = game_engine.take_action(&game_state, &action);
        analysis.push((action, metrics));
    }

    let terminal_score = game_engine
        .terminal_state(&game_state)
        .ok_or_else(|| anyhow!("Expected a terminal state"))?;

    Ok((SelfPlayMetrics::new(analysis, terminal_score), game_state))
}
