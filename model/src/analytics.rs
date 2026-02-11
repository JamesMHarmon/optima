use half::*;
use std::future::Future;

pub trait GameAnalyzer {
    type Action;
    type State;
    type Predictions;
    type Future: Future<Output = GameStateAnalysis<Self::Action, Self::Predictions>>;

    fn analyze_async(&self, game_state: &Self::State) -> Self::Future;

    fn analyze(
        &self,
        game_state: &Self::State,
    ) -> GameStateAnalysis<Self::Action, Self::Predictions>;

    fn prefetch(&self, game_state: &Self::State);
}

#[derive(Clone, Debug)]
pub struct GameStateAnalysis<A, P> {
    policy_scores: Vec<ActionWithPolicy<A>>,
    predictions: P,
}

impl<A, P> GameStateAnalysis<A, P> {
    pub fn policy_scores(&self) -> &[ActionWithPolicy<A>] {
        &self.policy_scores
    }

    pub fn predictions(&self) -> &P {
        &self.predictions
    }

    pub fn into_inner(self) -> (Vec<ActionWithPolicy<A>>, P) {
        (self.policy_scores, self.predictions)
    }
}

impl<A, P> GameStateAnalysis<A, P> {
    pub fn new(policy_scores: Vec<ActionWithPolicy<A>>, predictions: P) -> Self {
        Self {
            policy_scores,
            predictions,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ActionWithPolicy<A> {
    action: A,
    policy_score: f16,
}

impl<A> ActionWithPolicy<A> {
    pub fn new(action: A, policy_score: f16) -> Self {
        ActionWithPolicy {
            action,
            policy_score,
        }
    }

    pub fn action(&self) -> &A {
        &self.action
    }

    pub fn policy_score(&self) -> f16 {
        self.policy_score
    }
}
