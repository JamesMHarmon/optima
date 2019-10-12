use std::fmt::Debug;
use serde::{Serialize, Deserialize};
use failure::{Error,format_err};

use engine::engine::GameEngine;
use engine::game_state::GameState;
use mcts::mcts::{DirichletOptions,MCTS,MCTSOptions};
use model::analytics::GameAnalyzer;
use model::node_metrics::NodeMetrics;

#[derive(Serialize, Deserialize, Debug)]
pub struct SelfPlayMetrics<A,V> {
    analysis: Vec<(A, NodeMetrics<A>)>,
    score: V
}

#[derive(Debug)]
pub struct SelfPlayOptions {
    pub temperature: f32,
    pub temperature_max_actions: usize,
    pub temperature_post_max_actions: f32,
    pub visits: usize,
    pub parallelism: usize,
    pub fpu: f32,
    pub fpu_root: f32,
    pub cpuct_base: f32,
    pub cpuct_init: f32,
    pub cpuct_root_scaling: f32,
    pub alpha: f32,
    pub epsilon: f32
}

impl<A,V> SelfPlayMetrics<A,V> {

    pub fn take(self) -> (Vec<(A, NodeMetrics<A>)>, V) {
        (self.analysis, self.score)
    }
}

#[allow(non_snake_case)]
pub async fn self_play<'a, S, A, E, M, V>(
    game_engine: &'a E,
    analytics: &'a M,
    options: &'a SelfPlayOptions
) -> Result<SelfPlayMetrics<A,V>, Error>
    where
    S: GameState,
    A: Clone + Eq + Debug,
    E: 'a + GameEngine<State=S,Action=A,Value=V>,
    M: 'a + GameAnalyzer<State=S,Action=A>
{
    let game_state: S = S::initial();
    let num_actions = 0;
    let cpuct_base = options.cpuct_base;
    let cpuct_init = options.cpuct_init;
    let cpuct_root_scaling = options.cpuct_root_scaling;

    let mut mcts = MCTS::with_capacity(
        game_state,
        num_actions,
        game_engine,
        analytics,
        MCTSOptions::<S,A,_,_>::new(
            Some(DirichletOptions {
                alpha: options.alpha,
                epsilon: options.epsilon
            }),
            0.0,
            1.0,
            |_,_,_,Nsb,is_root| (((Nsb as f32 + cpuct_base + 1.0) / cpuct_base).ln() + cpuct_init) * if is_root { cpuct_root_scaling } else { 1.0 },
            |_,num_actions| if num_actions < options.temperature_max_actions { options.temperature } else { options.temperature_post_max_actions },
            options.parallelism
        ),
        options.visits
    );

    let mut state: S = S::initial();
    let mut analysis = Vec::new();

    while game_engine.is_terminal_state(&state).is_none() {
        mcts.search(options.visits).await?;
        let action = mcts.select_action().await?;
        let metrics = mcts.get_root_node_metrics().await?;

        mcts.advance_to_action(action.to_owned()).await?;
        state = game_engine.take_action(&state, &action);
        analysis.push((action, metrics));
    };

    let score = game_engine.is_terminal_state(&state).ok_or(format_err!("Expected a terminal state"))?;

    Ok(SelfPlayMetrics::<A,V> {
        analysis,
        score
    })
}