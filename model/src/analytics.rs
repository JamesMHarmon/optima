use engine::value::Value;
use half::*;
use std::future::Future;

pub trait GameAnalyzer {
    type Future: Future<Output = Self::GameStateAnalysis>;
    type GameStateAnalysis: GameStateAnalysis<Self::Action, Self::Value>;
    type Action;
    type State;
    type Value: Value;

    fn get_state_analysis(&self, game_state: &Self::State) -> Self::Future;
}

#[derive(Clone, Debug)]
pub struct BasicGameStateAnalysis<A, V> {
    policy_scores: Vec<ActionWithPolicy<A>>,
    pub value_score: V,
    pub moves_left: f32,
}

impl<A, V> GameStateAnalysis<A, V> for BasicGameStateAnalysis<A, V> {
    fn value_score(&self) -> &V {
        &self.value_score
    }

    fn moves_left_score(&self) -> f32 {
        self.moves_left
    }

    fn policy_scores(&self) -> &[ActionWithPolicy<A>] {
        &self.policy_scores
    }

    fn into_inner(self) -> (Vec<ActionWithPolicy<A>>, V, f32) {
        (self.policy_scores, self.value_score, self.moves_left)
    }
}

impl<A, V> BasicGameStateAnalysis<A, V> {
    pub fn new(value_score: V, policy_scores: Vec<ActionWithPolicy<A>>, moves_left: f32) -> Self {
        Self {
            policy_scores,
            value_score,
            moves_left,
        }
    }
}

pub trait GameStateAnalysis<A, V> {
    fn value_score(&self) -> &V;
    fn moves_left_score(&self) -> f32;
    fn policy_scores(&self) -> &[ActionWithPolicy<A>];
    fn into_inner(self) -> (Vec<ActionWithPolicy<A>>, V, f32);
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
