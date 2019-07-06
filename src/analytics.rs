use std::future::Future;

pub trait GameAnalytics
{
    type Future: Future<Output=GameStateAnalysis<Self::Action>>;
    type Action;
    type State;

    fn get_state_analysis(&self, game_state: &Self::State) -> Self::Future;
}

#[derive(Clone)]
pub struct GameStateAnalysis<A> {
    pub policy_scores: Vec<ActionWithPolicy<A>>,
    pub value_score: f64
}

impl<A> GameStateAnalysis<A> {
    pub fn new(value_score: f64, policy_scores: Vec<ActionWithPolicy<A>>) -> Self {
        GameStateAnalysis {
            policy_scores,
            value_score
        }
    }
}

#[derive(Clone)]
pub struct ActionWithPolicy<A> {
    pub action: A,
    pub policy_score: f64,
}

impl<A> ActionWithPolicy<A> {
    pub fn new(action: A, policy_score: f64) -> Self {
        ActionWithPolicy {
            action,
            policy_score
        }
    }
}
