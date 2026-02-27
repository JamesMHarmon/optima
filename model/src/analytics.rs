use half::*;

pub type RequestId = u64;

pub trait GameAnalyzer {
    type Action;
    type State;
    type Predictions;

    fn analyze(&self, request_id: RequestId, game_state: &Self::State);

    fn recv(
        &self,
    ) -> (
        RequestId,
        GameStateAnalysis<Self::Action, Self::Predictions>,
    );
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

    pub fn set_policy_score(&mut self, policy_score: f16) {
        self.policy_score = policy_score;
    }
}
