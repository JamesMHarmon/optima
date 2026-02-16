use engine::engine::GameEngine;
use engine::game_state::GameState;
use half::f16;
use model::{
    GameStateAnalysis,
    analytics::{ActionWithPolicy, GameAnalyzer},
};

use crate::GameLength;

#[derive(Hash, PartialEq, Eq, Clone, Debug)]
pub struct CountingGameState {
    pub p1_turn: bool,
    pub count: usize,
}

impl CountingGameState {
    pub fn is_terminal_state(&self) -> Option<CountingGamePredictions> {
        if self.count == 100 {
            Some(CountingGamePredictions([1.0, 0.0]))
        } else if self.count == 0 {
            Some(CountingGamePredictions([0.0, 1.0]))
        } else {
            None
        }
    }
}

impl GameState for CountingGameState {
    fn initial() -> Self {
        Self {
            p1_turn: true,
            count: 50,
        }
    }
}

pub struct CountingGameEngine {}

impl CountingGameEngine {
    #[cfg(test)]
    pub fn new() -> Self {
        Self {}
    }
}

#[derive(Clone)]
pub struct CountingGamePredictions(pub [f32; 2]);

impl engine::value::Value for CountingGamePredictions {
    fn get_value_for_player(&self, player: usize) -> f32 {
        self.0[player - 1]
    }
}

impl GameLength for CountingGamePredictions {
    fn game_length_score(&self) -> f32 {
        0.0
    }
}

#[derive(PartialEq, Eq, Clone, Debug)]
pub enum CountingAction {
    Increment,
    Decrement,
    Stay,
}

impl GameEngine for CountingGameEngine {
    type Action = CountingAction;
    type State = CountingGameState;
    type Terminal = CountingGamePredictions;

    fn take_action(&self, game_state: &Self::State, action: &Self::Action) -> Self::State {
        let count = game_state.count;

        let new_count = match action {
            CountingAction::Increment => count + 1,
            CountingAction::Decrement => count - 1,
            CountingAction::Stay => count,
        };

        Self::State {
            p1_turn: !game_state.p1_turn,
            count: new_count,
        }
    }

    fn terminal_state(&self, game_state: &Self::State) -> Option<Self::Terminal> {
        game_state.is_terminal_state()
    }

    fn player_to_move(&self, game_state: &Self::State) -> usize {
        if game_state.p1_turn { 1 } else { 2 }
    }

    fn move_number(&self, _game_state: &Self::State) -> usize {
        0
    }
}

pub struct CountingAnalyzer {
    policy_scores: [f16; 3],
}

impl CountingAnalyzer {
    #[cfg(test)]
    pub fn new(policy_scores: [f32; 3]) -> Self {
        let policy_scores = policy_scores
            .into_iter()
            .map(f16::from_f32)
            .collect::<Vec<f16>>();

        Self {
            policy_scores: [policy_scores[0], policy_scores[1], policy_scores[2]],
        }
    }
}

impl GameAnalyzer for CountingAnalyzer {
    type Action = CountingAction;
    type State = CountingGameState;
    type Predictions = CountingGamePredictions;

    fn analyze(
        &self,
        game_state: &Self::State,
    ) -> GameStateAnalysis<Self::Action, Self::Predictions> {
        let count = game_state.count as f32;

        if let Some(value_score) = game_state.is_terminal_state() {
            return GameStateAnalysis::new(Vec::new(), value_score);
        }

        let value_score = CountingGamePredictions([count / 100.0, (100.0 - count) / 100.0]);
        let policy_scores = vec![
            ActionWithPolicy::new(CountingAction::Increment, self.policy_scores[0]),
            ActionWithPolicy::new(CountingAction::Decrement, self.policy_scores[1]),
            ActionWithPolicy::new(CountingAction::Stay, self.policy_scores[2]),
        ];

        GameStateAnalysis::new(policy_scores, value_score)
    }

    fn prefetch(&self, _game_state: &Self::State) {
        // No-op for this simple test analyzer.
    }
}
