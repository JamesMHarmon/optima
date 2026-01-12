use crate::Predictions;

use super::MAX_NUMBER_OF_MOVES;
use super::{Action, GameState};
use engine::{PlayerResult, PlayerScore, Players, ValidActions};

pub struct Engine {}

impl Engine {
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for Engine {
    fn default() -> Self {
        Self::new()
    }
}

impl engine::engine::GameEngine for Engine {
    type Action = Action;
    type State = GameState;
    type Terminal = Predictions;

    fn take_action(&self, game_state: &Self::State, action: &Self::Action) -> Self::State {
        game_state.take_action(action)
    }

    fn player_to_move(&self, game_state: &Self::State) -> usize {
        game_state.player_to_move()
    }

    fn move_number(&self, game_state: &Self::State) -> usize {
        game_state.get_move_number()
    }

    fn terminal_state(&self, game_state: &Self::State) -> Option<Self::Terminal> {
        let value = game_state.is_terminal().map(|v| v.into()).or_else(|| {
            if game_state.get_move_number() > MAX_NUMBER_OF_MOVES {
                Some([0.0, 0.0].into())
            } else {
                None
            }
        });

        value.map(|value| Predictions::new(value, game_state.get_move_number() as f32))
    }
}

impl ValidActions for Engine {
    type Action = Action;
    type State = GameState;

    fn valid_actions(&self, game_state: &Self::State) -> impl Iterator<Item = Self::Action> {
        game_state.valid_actions().into_iter()
    }
}

impl Players for Engine {
    type State = GameState;

    fn players(&self, _state: &Self::State) -> &[usize] {
        static PLAYERS: [usize; 2] = [1, 2];
        &PLAYERS
    }
}

impl PlayerScore for Engine {
    type State = GameState;
    type PlayerScore = usize;

    fn score(&self, state: &Self::State, player_id: usize) -> Option<Self::PlayerScore> {
        state
            .is_terminal()
            .map(|value| if value.0[player_id - 1] > 0.5 { 1 } else { 0 })
    }
}

impl PlayerResult for Engine {
    type State = GameState;
    type PlayerResult = &'static str;

    fn result(&self, state: &Self::State, player_id: usize) -> Option<Self::PlayerResult> {
        state
            .is_terminal()
            .map(|value| match value.0[player_id - 1] {
                v if v > 0.5 => "win",
                v if v < 0.5 => "loss",
                _ => "draw",
            })
    }
}
