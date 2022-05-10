use super::MAX_NUMBER_OF_MOVES;
use super::{Action, GameState, Value};

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
    type Value = Value;

    fn take_action(&self, game_state: &Self::State, action: &Self::Action) -> Self::State {
        game_state.take_action(action)
    }

    fn get_player_to_move(&self, game_state: &Self::State) -> usize {
        if game_state.is_p1_turn_to_move() {
            1
        } else {
            2
        }
    }

    fn get_move_number(&self, game_state: &Self::State) -> usize {
        game_state.get_move_number()
    }

    fn is_terminal_state(&self, game_state: &Self::State) -> Option<Self::Value> {
        game_state.is_terminal().map(|v| v.into()).or_else(|| {
            if game_state.get_move_number() > MAX_NUMBER_OF_MOVES {
                Some([0.0, 0.0].into())
            } else {
                None
            }
        })
    }
}
