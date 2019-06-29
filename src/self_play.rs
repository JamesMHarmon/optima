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
    analysis: Vec<(A, NodeMetrics<A>)>,
    score: f64
}

pub struct SelfPlaySample<S, A> {
    pub game_state: S,
    pub score: f64,
    pub policy: NodeMetrics<A>
}

#[derive(Debug)]
pub struct SelfPlayOptions {
    pub temperature: f64,
    pub visits: usize,
    pub cpuct: f64,
    pub alpha: f64,
    pub epsilon: f64
}

impl<A> SelfPlayMetrics<A> {
    pub fn take_analysis(self) -> Vec<(A, NodeMetrics<A>)> {
        self.analysis
    }

    pub fn score(&self) -> f64 {
        self.score
    }
}

pub fn self_play<'a, S, A, E, M>(game_engine: &E, analytics: &M, options: &SelfPlayOptions) -> Result<SelfPlayMetrics<A>, &'static str>
    where
    S: GameState,
    A: Clone + Eq,
    E: 'a + GameEngine<State=S, Action=A>,
    M: 'a + GameAnalytics<State=S, Action=A>
{
    let uuid = Uuid::new_v4();
    let seedable_rng = rng::create_rng_from_uuid(uuid);
    let game_state: S = S::initial();
    let cpuct = options.cpuct;
    let temperature = options.temperature;

    let mut mcts = MCTS::new(
        game_state,
        game_engine,
        analytics,
        MCTSOptions::new(
            Some(DirichletOptions {
                alpha: options.alpha,
                epsilon: options.epsilon
            }),
            &|_,_| 4.0,
            &|_| 1.0,
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
        let search_result = mcts.search(options.visits)?;
        let action = search_result.0;
        let metrics = mcts.get_root_node_metrics()?;

        mcts.advance_to_action(&action)?;
        state = game_engine.take_action(&state, &action);
        self_play_metrics.analysis.push((action, metrics));
    };

    let final_score = game_engine.is_terminal_state(&state).unwrap();
    let p1_last_to_move = self_play_metrics.analysis.len() % 2 == 1;
    let final_score_p1 = if p1_last_to_move { final_score * -1.0 } else { final_score };
    self_play_metrics.score = final_score_p1;

    Ok(self_play_metrics)
}