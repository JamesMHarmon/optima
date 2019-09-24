use std::fmt::Debug;
use serde::{Serialize, Deserialize};
use uuid::Uuid;
use failure::{Error,format_err};

use common::rng;
use common::linked_list::List;
use engine::engine::GameEngine;
use engine::game_state::GameState;
use mcts::mcts::{DirichletOptions,MCTS,MCTSOptions};
use model::analytics::GameAnalyzer;
use model::node_metrics::NodeMetrics;

#[derive(Serialize, Deserialize, Debug)]
pub struct SelfPlayMetrics<A> {
    guid: String,
    analysis: Vec<(A, NodeMetrics<A>)>,
    score: f32
}

#[derive(Debug)]
pub struct SelfPlayOptions {
    pub temperature: f32,
    pub temperature_max_actions: usize,
    pub temperature_post_max_actions: f32,
    pub visits: usize,
    pub fpu: f32,
    pub fpu_root: f32,
    pub cpuct_base: f32,
    pub cpuct_init: f32,
    pub cpuct_root_scaling: f32,
    pub alpha: f32,
    pub epsilon: f32
}

impl<A> SelfPlayMetrics<A> {
    pub fn take_analysis(self) -> Vec<(A, NodeMetrics<A>)> {
        self.analysis
    }

    pub fn score(&self) -> f32 {
        self.score
    }
}

#[allow(non_snake_case)]
pub async fn self_play<'a, S, A, E, M>(
    game_engine: &'a E,
    analytics: &'a M,
    options: &'a SelfPlayOptions
) -> Result<SelfPlayMetrics<A>, Error>
    where
    S: GameState,
    A: Clone + Eq + Debug,
    E: 'a + GameEngine<State=S, Action=A>,
    M: 'a + GameAnalyzer<State=S, Action=A>
{
    let uuid = Uuid::new_v4();
    let seedable_rng = rng::create_rng_from_uuid(uuid);
    let game_state: S = S::initial();
    let actions = List::new();
    let cpuct_base = options.cpuct_base;
    let cpuct_init = options.cpuct_init;
    let cpuct_root_scaling = options.cpuct_root_scaling;

    let mut mcts = MCTS::new(
        game_state,
        actions,
        game_engine,
        analytics,
        MCTSOptions::<S,A,_,_,_>::new(
            Some(DirichletOptions {
                alpha: options.alpha,
                epsilon: options.epsilon
            }),
            0.0,
            1.0,
            |_,_,_,Nsb,is_root| (((Nsb as f32 + cpuct_base + 1.0) / cpuct_base).ln() + cpuct_init) * if is_root { cpuct_root_scaling } else { 1.0 },
            |_,actions| if actions.len() < options.temperature_max_actions { options.temperature } else { options.temperature_post_max_actions },
            seedable_rng,
        )
    );

    let mut state: S = S::initial();
    let mut self_play_metrics = SelfPlayMetrics::<A> {
        guid: format!("{}", uuid),
        analysis: Vec::new(),
        score: 0.0
    };

    while game_engine.is_terminal_state(&state) == None {
        mcts.search(options.visits).await?;
        let action = mcts.select_action().await?;
        let metrics = mcts.get_root_node_metrics().await?;

        mcts.advance_to_action(action.to_owned()).await?;
        state = game_engine.take_action(&state, &action);
        self_play_metrics.analysis.push((action, metrics));
    };

    let final_score = game_engine.is_terminal_state(&state).ok_or(format_err!("Expected a terminal state"))?;
    let p1_last_to_move = self_play_metrics.analysis.len() % 2 == 1;
    let final_score_p1 = if p1_last_to_move { final_score * -1.0 } else { final_score };
    self_play_metrics.score = final_score_p1;

    Ok(self_play_metrics)
}