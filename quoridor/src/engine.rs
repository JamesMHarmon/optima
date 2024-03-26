use super::{Action, GameState, Value};
use engine::engine::GameEngine;

#[derive(Default)]
pub struct Engine {}

impl Engine {
    pub fn new() -> Self {
        Self {}
    }
}

impl GameEngine for Engine {
    type Action = Action;
    type State = GameState;
    type Value = Value;

    fn take_action(&self, game_state: &Self::State, action: &Self::Action) -> Self::State {
        game_state.take_action(action)
    }

    fn is_terminal_state(&self, game_state: &Self::State) -> Option<Self::Value> {
        game_state.is_terminal()
    }

    fn get_player_to_move(&self, game_state: &Self::State) -> usize {
        if game_state.p1_turn_to_move {
            1
        } else {
            2
        }
    }

    fn get_move_number(&self, game_state: &Self::State) -> usize {
        game_state.num_moves / 2 + 1
    }
}
