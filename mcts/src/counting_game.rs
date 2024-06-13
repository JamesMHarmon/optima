use engine::engine::GameEngine;
use engine::game_state::GameState;
use futures::future;
use half::f16;
use model::{
    analytics::{ActionWithPolicy, GameAnalyzer},
    GameStateAnalysis,
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
        if game_state.p1_turn {
            1
        } else {
            2
        }
    }

    fn move_number(&self, _game_state: &Self::State) -> usize {
        0
    }
}

pub struct CountingAnalyzer {}

impl CountingAnalyzer {
    #[cfg(test)]
    pub fn new() -> Self {
        Self {}
    }
}

impl GameAnalyzer for CountingAnalyzer {
    type Action = CountingAction;
    type State = CountingGameState;
    type Future = future::Ready<GameStateAnalysis<Self::Action, Self::Predictions>>;
    type Predictions = CountingGamePredictions;

    fn get_state_analysis(&self, game_state: &Self::State) -> Self::Future {
        let count = game_state.count as f32;

        if let Some(value_score) = game_state.is_terminal_state() {
            return future::ready(GameStateAnalysis::new(Vec::new(), value_score));
        }

        let value_score = CountingGamePredictions([count / 100.0, (100.0 - count) / 100.0]);
        let policy_scores = vec![
            ActionWithPolicy {
                action: CountingAction::Increment,
                policy_score: f16::from_f32(0.3),
            },
            ActionWithPolicy {
                action: CountingAction::Decrement,
                policy_score: f16::from_f32(0.3),
            },
            ActionWithPolicy {
                action: CountingAction::Stay,
                policy_score: f16::from_f32(0.4),
            },
        ];

        future::ready(GameStateAnalysis::new(policy_scores, value_score))
    }
}
