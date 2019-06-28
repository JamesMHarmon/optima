use serde::{Serialize, Deserialize};
use uuid::Uuid;

use super::game_state::GameState;
use super::analytics::GameAnalytics;
use super::rng;
use super::mcts::{DirichletOptions,MCTS,MCTSOptions};
use super::node_metrics::{NodeMetrics};
use super::engine::GameEngine;

#[derive(Serialize, Deserialize, Debug)]
pub struct SelfPlayMetrics<A> {
    guid: String,
    analysis: Vec<(A, NodeMetrics<A>)>
}

pub fn self_play<'a, S, A, E, M>(game_engine: &E, analytics: &M) -> Result<SelfPlayMetrics<A>, &'static str>
    where
    S: GameState,
    A: Clone + Eq,
    E: 'a + GameEngine<State=S, Action=A>,
    M: 'a + GameAnalytics<State=S, Action=A>
{
    let uuid = Uuid::new_v4();
    let seedable_rng = rng::create_rng_from_uuid(uuid);
    let game_state: S = S::initial();

    let mut mcts = MCTS::new(
        game_state,
        game_engine,
        analytics,
        MCTSOptions::new(
            Some(DirichletOptions {
                alpha: 0.3,
                epsilon: 0.25
            }),
            &|_,_| 4.0,
            &|_| 1.0,
            seedable_rng,
        )
    );

    let mut state: S = S::initial();
    let mut self_play_metrics = SelfPlayMetrics::<A> {
        guid: format!("{}", uuid),
        analysis: Vec::new()
    };

    while game_engine.is_terminal_state(&state) == None {
        let search_result = mcts.search(5)?;
        let action = search_result.0;
        let metrics = mcts.get_root_node_metrics()?;

        mcts.advance_to_action(&action)?;
        state = game_engine.take_action(&state, &action);
        self_play_metrics.analysis.push((action, metrics));
    };

    Ok(self_play_metrics)
}