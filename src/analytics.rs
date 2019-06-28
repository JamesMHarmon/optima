pub trait GameAnalytics {
    type Action;
    type State;

    fn get_state_analysis(&self, game_state: &Self::State) -> GameStateAnalysis<Self::Action>;
}

#[derive(Clone)]
pub struct GameStateAnalysis<A> {
    pub policy_scores: Vec<ActionWithPolicy<A>>,
    pub value_score: f64
}

impl<A> GameStateAnalysis<A> {
    pub fn new(policy_scores: Vec<ActionWithPolicy<A>>, value_score: f64) -> Self {
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
