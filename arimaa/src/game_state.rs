use anyhow::Result;
use std::fmt::{self, Display};
use std::{ops::Deref, str::FromStr};

use arimaa_engine::{Action, GameState as ArimaaGameState, Piece, PieceBoard, Value};

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

    pub fn get_piece_board(&self) -> &PieceBoard {
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

    // Returns either 1 or 2 depending on the player to move.
    pub fn player_to_move(&self) -> usize {
        if self.0.is_p1_turn_to_move() {
            1
        } else {
            2
        }
    }

    pub fn get_vertical_symmetry(&self) -> Self {
        Self(self.0.get_vertical_symmetry())
    }

    pub fn piece_to_place(&self) -> Piece {
        self.0.piece_to_place()
    }
}

impl Display for GameState {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        ArimaaGameState::fmt(&self.0, fmt)
    }
}

impl FromStr for GameState {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        Ok(GameState(ArimaaGameState::from_str(s)?))
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
