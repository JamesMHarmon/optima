use engine::game_state::GameState;
use engine::engine::GameEngine;
use model::analytics::{ActionWithPolicy,GameAnalyzer,GameStateAnalysis};
use futures::future;

#[derive(Hash, PartialEq, Eq, Clone, Debug)]
pub struct CountingGameState {
    pub p1_turn: bool,
    pub count: usize
}

impl CountingGameState {
    #[cfg(test)]
    pub fn from_starting_count(p1_turn: bool, count: usize) -> Self {
        Self { p1_turn, count }
    }

    pub fn is_terminal_state(&self) -> Option<[f32; 2]> {
        if self.count == 100 {
            Some([1.0, 0.0])
        } else if self.count == 0 {
            Some([0.0, 1.0])
        } else {
            None
        }
    }
}

impl GameState for CountingGameState {
    fn initial() -> Self {
        Self { p1_turn: true, count: 50 }
    }
}

pub struct CountingGameEngine {

}

impl CountingGameEngine {
    #[cfg(test)]
    pub fn new() -> Self { Self {} }
}

#[derive(PartialEq, Eq, Clone, Debug)]
pub enum CountingAction {
    Increment,
    Decrement,
    Stay
}

impl GameEngine for CountingGameEngine {
    type Action = CountingAction;
    type State = CountingGameState;
    type Value = [f32; 2];

    fn take_action(&self, game_state: &Self::State, action: &Self::Action) -> Self::State {
        let count = game_state.count;

        let new_count = match action {
            CountingAction::Increment => count + 1,
            CountingAction::Decrement => count - 1,
            CountingAction::Stay => count
        };

        Self::State { p1_turn: !game_state.p1_turn, count: new_count }
    }

    fn is_terminal_state(&self, game_state: &Self::State) -> Option<Self::Value> {
        game_state.is_terminal_state()
    }

    fn get_player_to_move(&self, game_state: &Self::State) -> usize {
        if game_state.p1_turn { 1 } else { 2 }
    }
}

pub struct CountingAnalyzer {

}

impl CountingAnalyzer {
    #[cfg(test)]
    pub fn new() -> Self { Self {} }
}

impl GameAnalyzer for CountingAnalyzer {
    type Action = CountingAction;
    type State = CountingGameState;
    type Future = future::Ready<GameStateAnalysis<Self::Action,Self::Value>>;
    type Value = [f32; 2];

    fn get_state_analysis(&self, game_state: &Self::State) -> Self::Future {
        let count = game_state.count as f32;

        if let Some(score) = game_state.is_terminal_state() {
            return future::ready(GameStateAnalysis {
                policy_scores: Vec::new(),
                value_score: score
            });
        }
        
        future::ready(GameStateAnalysis {
            policy_scores: vec!(
                ActionWithPolicy {
                    action: CountingAction::Increment,
                    policy_score: 0.3
                },
                ActionWithPolicy {
                    action: CountingAction::Decrement,
                    policy_score: 0.3
                },
                ActionWithPolicy {
                    action: CountingAction::Stay,
                    policy_score: 0.4
                },
            ),
            value_score: [(count as f32) / 100.0, (100.0 - count as f32) / 100.0]
        })
    }

    fn get_value_for_player_to_move(&self, game_state: &Self::State, value: &Self::Value) -> f32 {
        value[if game_state.p1_turn { 0 } else { 1 }]
    }

    fn get_value_for_player(&self, player: usize, value: &Self::Value) -> f32 {
        value[player - 1]
    }
}
