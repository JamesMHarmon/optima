use arimaa_engine::{Action, PieceBoard, Value};
use std::ops::Deref;

#[derive(Hash, Debug, Clone)]
pub struct GameState(arimaa_engine::GameState);

impl GameState {
    pub fn new(p1_turn_to_move: bool, move_number: usize, piece_board: PieceBoard) -> Self {
        GameState(arimaa_engine::GameState::new(
            p1_turn_to_move,
            move_number,
            piece_board,
        ))
    }

    #[must_use]
    pub fn take_action(&self, action: &Action) -> Self {
        Self(self.0.take_action(action))
    }

    pub fn is_terminal(&self) -> Option<Value> {
        self.0.is_terminal()
    }

    pub fn valid_actions(&self) -> Vec<Action> {
        self.0.valid_actions()
    }

    pub fn valid_actions_no_exclusions(&self) -> Vec<Action> {
        self.0.valid_actions_no_exclusions()
    }

    pub fn get_piece_board(&self) -> PieceBoard {
        self.0.get_piece_board()
    }

    pub fn is_p1_turn_to_move(&self) -> bool {
        self.0.is_p1_turn_to_move()
    }

    pub fn get_move_number(&self) -> usize {
        self.0.get_move_number()
    }

    pub fn get_transposition_hash(&self) -> u64 {
        self.0.hash().board_state_hash()
    }

    pub fn step(&self) -> usize {
        self.0.step()
    }

    pub fn is_play_phase(&self) -> bool {
        self.0.is_play_phase()
    }
}

impl engine::GameState for GameState {
    fn initial() -> Self {
        GameState(arimaa_engine::GameState::initial())
    }
}

impl Deref for GameState {
    type Target = arimaa_engine::GameState;

    fn deref(&self) -> &<Self as Deref>::Target {
        &self.0
    }
}
