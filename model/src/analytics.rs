use half::*;
use std::future::Future;

pub trait GameAnalyzer {
    type Action;
    type State;
    type Predictions;
    type Future: Future<Output = GameStateAnalysis<Self::Action, Self::Predictions>>;

    fn get_state_analysis(&self, game_state: &Self::State) -> Self::Future;
}

#[derive(Clone, Debug)]
pub struct GameStateAnalysis<A, P> {
    policy_scores: Vec<ActionWithPolicy<A>>,
    predictions: P
}

impl<A, P> GameStateAnalysis<A, P> {
    fn policy_scores(&self) -> &[ActionWithPolicy<A>] {
        &self.policy_scores
    }

    fn predictions(&self) -> &P {
        &self.predictions
    }

    fn into_inner(self) -> (Vec<ActionWithPolicy<A>>, P) {
        (self.policy_scores, self.predictions)
    }
}

impl<A, P> GameStateAnalysis<A, P> {
    pub fn new(policy_scores: Vec<ActionWithPolicy<A>>, predictions: P) -> Self {
        Self {
            policy_scores,
            predictions
        }
    }
}

#[derive(Clone, Debug)]
pub struct ActionWithPolicy<A> {
    pub action: A,
    pub policy_score: f16,
}

impl<A> ActionWithPolicy<A> {
    pub fn new(action: A, policy_score: f16) -> Self {
        ActionWithPolicy {
            action,
            policy_score,
        }
    }
}
