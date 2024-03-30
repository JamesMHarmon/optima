use engine::value::Value;
use std::future::Future;

pub trait GameAnalyzer {
    type Future: Future<Output = Self::GameStateAnalytics>;
    type GameStateAnalytics: GameStateAnalytics<Self::Action, Self::Value>;
    type Action;
    type State;
    type Value: Value;

    fn get_state_analysis(&self, game_state: &Self::State) -> Self::Future;
}

#[derive(Clone, Debug)]
pub struct GameStateAnalysis<A, V> {
    policy_scores: Vec<ActionWithPolicy<A>>,
    pub value_score: V,
    pub moves_left: f32,
}

impl<A, V> GameStateAnalytics<A, V> for GameStateAnalysis<A, V> {
    fn value_score(&self) -> &V {
        &self.value_score
    }

    fn moves_left_score(&self) -> f32 {
        self.moves_left
    }

    fn policy_scores(&self) -> &[ActionWithPolicy<A>] {
        &self.policy_scores
    }
}

impl<A, V> GameStateAnalysis<A, V> {
    pub fn new(value_score: V, policy_scores: Vec<ActionWithPolicy<A>>, moves_left: f32) -> Self {
        GameStateAnalysis {
            policy_scores,
            value_score,
            moves_left,
        }
    }

    pub fn into_inner(self) -> (Vec<ActionWithPolicy<A>>, V, f32) {
        (self.policy_scores, self.value_score, self.moves_left)
    }
}

pub trait GameStateAnalytics<A, V> {
    fn value_score(&self) -> &V;
    fn moves_left_score(&self) -> f32;
    fn policy_scores(&self) -> &[ActionWithPolicy<A>];
}

#[derive(Clone, Debug)]
pub struct ActionWithPolicy<A> {
    pub action: A,
    pub policy_score: f32,
}

impl<A> ActionWithPolicy<A> {
    pub fn new(action: A, policy_score: f32) -> Self {
        ActionWithPolicy {
            action,
            policy_score,
        }
    }
}
