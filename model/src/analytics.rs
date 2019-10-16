use std::future::Future;
use engine::value::Value;

pub trait GameAnalyzer
{
    type Future: Future<Output=GameStateAnalysis<Self::Action,Self::Value>>;
    type Action;
    type State;
    type Value: Value;

    fn get_state_analysis(&self, game_state: &Self::State) -> Self::Future;
}

#[derive(Clone,Debug)]
pub struct GameStateAnalysis<A,V> {
    pub policy_scores: Vec<ActionWithPolicy<A>>,
    pub value_score: V
}

impl<A,V> GameStateAnalysis<A,V> {
    pub fn new(value_score: V, policy_scores: Vec<ActionWithPolicy<A>>) -> Self {
        GameStateAnalysis {
            policy_scores,
            value_score
        }
    }
}

#[derive(Clone,Debug)]
pub struct ActionWithPolicy<A> {
    pub action: A,
    pub policy_score: f32,
}

impl<A> ActionWithPolicy<A> {
    pub fn new(action: A, policy_score: f32) -> Self {
        ActionWithPolicy {
            action,
            policy_score
        }
    }
}
