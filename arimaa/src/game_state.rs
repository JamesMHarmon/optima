use arimaa_engine::{Action, GameState as ArimaaGameState, PieceBoard, Value};
use std::{fmt::{Display, self}, ops::Deref};

#[derive(Hash, Debug, Clone)]
pub struct GameState(ArimaaGameState);

impl GameState {
    pub fn new(p1_turn_to_move: bool, move_number: usize, piece_board: PieceBoard) -> Self {
        GameState(ArimaaGameState::new(
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

impl Display for GameState {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        ArimaaGameState::fmt(&self.0, fmt)
    }
}

impl engine::game_state::GameState for GameState {
    fn initial() -> Self {
        GameState(ArimaaGameState::initial())
    }
}

impl Deref for GameState {
    type Target = ArimaaGameState;

    fn deref(&self) -> &<Self as Deref>::Target {
        &self.0
    }
}
