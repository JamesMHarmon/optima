use anyhow::{Result, anyhow};
use rand::Rng;
use std::fmt::Debug;
use std::hash::Hash;

use common::{GameLength, PlayerToMove, TranspositionHash};
use engine::{GameEngine, GameState, ValidActions};
use model::GameAnalyzer;
use puct::{
    DirichletOptions, NoTemp, PuctMCTS, RollupStats, SelectionPolicy, TemperatureConstant,
    ValueModel,
};

use super::{SelfPlayMetrics, SelfPlayOptions};

type SnapshotOf<VM> = <<VM as ValueModel>::Rollup as RollupStats>::Snapshot;
type SSOf<VM> = SnapshotOf<VM>;

#[allow(non_snake_case)]
pub async fn play_self_one<S, A, E, M, P, VM, Sel>(
    game_engine: &E,
    analyzer: &M,
    value_model: &VM,
    selection: &Sel,
    options: &SelfPlayOptions,
) -> Result<(SelfPlayMetrics<A, P, SSOf<VM>>, S)>
where
    S: GameState + Clone + TranspositionHash + PlayerToMove + Send + Sync,
    A: Clone + Eq + Hash + Debug + Send + Sync,
    E: GameEngine<State = S, Action = A, Terminal = P>,
    E: ValidActions<State = S, Action = A>,
    E: Sync,
    M: GameAnalyzer<State = S, Action = A, Predictions = P> + Sync,
    P: Clone + GameLength,
    VM: ValueModel<Predictions = P, Terminal = P> + Sync,
    Sel: SelectionPolicy<SnapshotOf<VM>, State = S, Action = A, Terminal = P> + Sync,
    <VM as ValueModel>::Rollup: Send + Sync,
    SnapshotOf<VM>: Clone + Send + Sync,
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
        play_options.parallelism,
        play_options.sim_threads,
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

        let metrics = mcts.node_metrics();

        mcts.advance_to_action(action.to_owned()).await?;
        game_state = game_engine.take_action(&game_state, &action);
        analysis.push((action, metrics));
    }

    let terminal_score = game_engine
        .terminal_state(&game_state)
        .ok_or_else(|| anyhow!("Expected a terminal state"))?;

    Ok((SelfPlayMetrics::new(analysis, terminal_score), game_state))
}
