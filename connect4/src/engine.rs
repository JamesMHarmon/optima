use std::fmt::{self,Display,Formatter};
use engine::engine::GameEngine;
use engine::game_state;

use super::action::Action;
use super::board::map_board_to_arr;

const TOP_ROW_MASK: u64 = 0b0100000_0100000_0100000_0100000_0100000_0100000_0100000;

#[derive(Hash, Eq, PartialEq, Clone, Debug)]
pub struct GameState {
    pub p1_turn_to_move: bool,
    pub p1_piece_board: u64,
    pub p2_piece_board: u64
}

impl game_state::GameState for GameState {
    fn initial() -> Self {
        GameState {
            p1_turn_to_move: true,
            p1_piece_board: 0,
            p2_piece_board: 0
        }
    }
}

impl GameState {
    pub fn drop_piece(&self, column: usize) -> Self {
        let column_adder = 1 << 7 * (column - 1);
        let all_pieces = self.p1_piece_board | self.p2_piece_board;
        let dropped_piece = (all_pieces + column_adder) & !all_pieces;
        let p1_turn_to_move = self.p1_turn_to_move;
        let mut p1_piece_board = self.p1_piece_board;
        let mut p2_piece_board = self.p2_piece_board;

        if p1_turn_to_move {
            p1_piece_board = self.p1_piece_board | dropped_piece;
        } else {
            p2_piece_board = self.p2_piece_board | dropped_piece;
        }

        Self {
            p1_turn_to_move: !p1_turn_to_move,
            p1_piece_board,
            p2_piece_board
        }
    }

    pub fn get_valid_actions(&self) -> Vec<bool> {
        let all_pieces = self.p1_piece_board | self.p2_piece_board;

        let valid_columns: Vec<bool> = (1..8).map(|column| {
            let column_mask = 1 << 7 * (column - 1);
            let column_mask_row_six = column_mask << 5;
            let is_column_full = column_mask_row_six & all_pieces != 0;
            !is_column_full
        }).collect();

        valid_columns
    }

    /// Determines if the current state is either a winning or drawn position.
    /// 
    /// Win: If the position is winning then this method will return Some(-1.0) since the value of the position
    /// is always from the reference of the current player to move, who just lost.
    /// 
    /// Drawn: If the position is a draw then the return will be Some(0.0);
    /// 
    /// Not Terminal: If the position is not yet the end of the game then None will be returned.
    pub fn is_terminal(&self) -> Option<f32> {
        let all_pieces = self.p1_piece_board | self.p2_piece_board;

        if self.has_connected_4() {
            return Some(-1.0);
        }

        if all_pieces & TOP_ROW_MASK == TOP_ROW_MASK {
            return Some(0.0);
        }

        None
    }

    pub fn number_of_actions(&self) -> usize {
        let mut all_pieces = self.p1_piece_board | self.p2_piece_board;
        let mut num_of_set_bits = 0;

        while all_pieces != 0 {
            all_pieces = all_pieces & (all_pieces - 1);
            num_of_set_bits += 1;
        }

        num_of_set_bits
    }

    fn has_connected_4(&self) -> bool {
        let board = if self.p1_turn_to_move { self.p2_piece_board } else { self.p1_piece_board };

        let c2 = board & (board << 6);
        if c2 & (c2 << (2 * 6)) != 0 {
            return true;
        }

        let c2 = board & (board << 7);
        if c2 & (c2 << (2 * 7)) != 0 {
            return true;
        }

        let c2 = board & (board << 8);
        if c2 & (c2 << (2 * 8)) != 0 {
            return true;
        }

        let c2 = board & (board << 1);
        if c2 & (c2 << 2) != 0 {
            return true;
        }

        return false;
    }
}

impl Display for GameState {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let p1_board = map_board_to_arr(self.p1_piece_board);
        let p2_board = map_board_to_arr(self.p2_piece_board);

        writeln!(f, "")?;
        writeln!(f, "-----------------------------")?;

        for y in 0..6 {
            for x in 0..7 {
                if x == 0 { write!(f, "|")?; }
                let idx = y * 7 + x;
                let p = if p1_board[idx] != 0.0 { "X" } else if p2_board[idx] != 0.0 { "O" } else { " " };
                write!(f, " {} |", p)?;
            }
            writeln!(f, "")?;
            if y != 5 { writeln!(f, "|---------------------------|")?; }
        }

        writeln!(f, "-----------------------------")?;

        Ok(())
    }
}

pub struct Engine {}

impl Engine {
    pub fn new() -> Self { Self {} }
}

impl GameEngine for Engine {
    type Action = Action;
    type State = GameState;

    fn take_action(&self, game_state: &GameState, action: &Action) -> GameState {
        match action {
            Action::DropPiece(column) => game_state.drop_piece(*column as usize)
        }
    }

    fn is_terminal_state(&self, game_state: &GameState) -> Option<f32> {
        game_state.is_terminal()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_state_is_correct() {
        let state: GameState = game_state::GameState::initial();
        assert_eq!(state.p1_turn_to_move, true);
        assert_eq!(state.p1_piece_board, 0);
        assert_eq!(state.p2_piece_board, 0);
    }

    #[test]
    fn test_drop_piece_switches_player() {
        let mut state: GameState = game_state::GameState::initial();
        state = state.drop_piece(1);
        assert_eq!(state.p1_turn_to_move, false);
        state = state.drop_piece(1);
        assert_eq!(state.p1_turn_to_move, true);
    }

    #[test]
    fn test_drop_piece_empty_first_column() {
        let mut state: GameState = game_state::GameState::initial();
        state = state.drop_piece(1);
        assert_eq!(state.p1_piece_board, 1);
    }

    #[test]
    fn test_drop_piece_empty_last_column() {
        let mut state: GameState = game_state::GameState::initial();
        state = state.drop_piece(7);
        assert_eq!(state.p1_piece_board, 1 << 7 * 6);
    }

    #[test]
    fn test_drop_piece_empty_column_includes_other_pieces() {
        let mut state: GameState = game_state::GameState::initial();
        state = state.drop_piece(1);
        state = state.drop_piece(2);
        state = state.drop_piece(3);

        let piece_1 = 1;
        let piece_3 = 1 << 7 * 2;
        assert_eq!(state.p1_piece_board, piece_1 | piece_3);
    }

    #[test]
    fn test_drop_piece_column_on_other_player_piece() {
        let mut state: GameState = game_state::GameState::initial();
        state = state.drop_piece(1);
        state = state.drop_piece(1);
        state = state.drop_piece(4);
        state = state.drop_piece(4);
        state = state.drop_piece(4);
        state = state.drop_piece(4);
        state = state.drop_piece(4);

        let piece_1_1 = 1;
        let piece_1_2 = 2;

        let column_4 = 1 << 7 * 3;
        let piece_4_1 = column_4;
        let piece_4_2 = column_4 << 1;
        let piece_4_3 = column_4 << 2;
        let piece_4_4 = column_4 << 3;
        let piece_4_5 = column_4 << 4;

        assert_eq!(state.p1_piece_board, piece_1_1 | piece_4_1 | piece_4_3 | piece_4_5);
        assert_eq!(state.p2_piece_board, piece_1_2 | piece_4_2 | piece_4_4);
    }

    #[test]
    fn test_get_valid_actions_all_columns_available() {
        let mut state: GameState = game_state::GameState::initial();

        for column in 1..8 {
            for _ in 1..6 {
                state = state.drop_piece(column);
            }
        }

        assert_eq!(state.get_valid_actions().as_slice(), [true, true, true, true, true, true, true]);
    }

    #[test]
    fn test_get_valid_actions_all_columns_full() {
        let mut state: GameState = game_state::GameState::initial();

        for column in 1..8 {
            for _ in 1..7 {
                state = state.drop_piece(column);
            }
        }

        assert_eq!(state.get_valid_actions().as_slice(), [false, false, false, false, false, false, false]);
    }

    #[test]
    fn test_get_valid_actions_first_column_full() {
        let mut state: GameState = game_state::GameState::initial();

        for column in 1..8 {
            for _ in 1..6 {
                state = state.drop_piece(column);
            }
        }

        state = state.drop_piece(1);

        assert_eq!(state.get_valid_actions().as_slice(), [false, true, true, true, true, true, true]);
    }

    #[test]
    fn test_get_valid_actions_last_column_full() {
        let mut state: GameState = game_state::GameState::initial();

        for column in 1..8 {
            for _ in 1..6 {
                state = state.drop_piece(column);
            }
        }

        state = state.drop_piece(7);

        assert_eq!(state.get_valid_actions().as_slice(), [true, true, true, true, true, true, false]);
    }

    #[test]
    fn test_get_valid_actions_three_columns_full() {
        let mut state: GameState = game_state::GameState::initial();

        for column in 1..8 {
            for _ in 1..6 {
                state = state.drop_piece(column);
            }
        }

        state = state.drop_piece(3);
        state = state.drop_piece(4);
        state = state.drop_piece(5);

        assert_eq!(state.get_valid_actions().as_slice(), [true, true, false, false, false, true, true]);
    }

    #[test]
    fn test_number_of_actions_none() {
        let state: GameState = game_state::GameState::initial();

        assert_eq!(state.number_of_actions(), 0);
    }

    #[test]
    fn test_number_of_actions_one() {
        let mut state: GameState = game_state::GameState::initial();
        state = state.drop_piece(3);

        assert_eq!(state.number_of_actions(), 1);
    }

    #[test]
    fn test_number_of_actions_two() {
        let mut state: GameState = game_state::GameState::initial();
        state = state.drop_piece(3);
        state = state.drop_piece(3);

        assert_eq!(state.number_of_actions(), 2);
    }

    #[test]
    fn test_number_of_actions_three() {
        let mut state: GameState = game_state::GameState::initial();
        state = state.drop_piece(3);
        state = state.drop_piece(3);
        state = state.drop_piece(4);

        assert_eq!(state.number_of_actions(), 3);
    }
}
