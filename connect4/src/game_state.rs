use std::{
    fmt::{self, Display, Formatter},
    hash::{Hash, Hasher},
};

use common::TranspositionHash;

use crate::{Value, map_board_to_arr};

use super::Zobrist;

const TOP_ROW_MASK: u64 = 0b0100000_0100000_0100000_0100000_0100000_0100000_0100000;

#[derive(Clone, Debug)]
pub struct GameState {
    pub p1_turn_to_move: bool,
    pub p1_piece_board: u64,
    pub p2_piece_board: u64,
    pub zobrist: Zobrist,
}

impl engine::GameState for GameState {
    fn initial() -> Self {
        GameState {
            p1_turn_to_move: true,
            p1_piece_board: 0,
            p2_piece_board: 0,
            zobrist: Zobrist::initial(),
        }
    }
}

impl Hash for GameState {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.zobrist.board_state_hash().hash(state);
    }
}

impl PartialEq for GameState {
    fn eq(&self, other: &Self) -> bool {
        self.zobrist.board_state_hash() == other.zobrist.board_state_hash()
    }
}

impl Eq for GameState {}

impl GameState {
    pub fn player_to_move(&self) -> usize {
        if self.p1_turn_to_move { 1 } else { 2 }
    }

    pub fn drop_piece(&self, column: usize) -> Self {
        let column_adder = 1 << (7 * (column - 1));
        let all_pieces = self.p1_piece_board | self.p2_piece_board;
        let dropped_piece = (all_pieces + column_adder) & !all_pieces;
        let p1_turn_to_move = self.p1_turn_to_move;
        let zobrist = self.zobrist.add_piece(dropped_piece, p1_turn_to_move);
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
            p2_piece_board,
            zobrist,
        }
    }

    pub fn get_valid_actions(&self) -> Vec<bool> {
        let all_pieces = self.p1_piece_board | self.p2_piece_board;

        let valid_columns: Vec<bool> = (1..8)
            .map(|column| {
                let column_mask = 1 << (7 * (column - 1));
                let column_mask_row_six = column_mask << 5;
                let is_column_full = column_mask_row_six & all_pieces != 0;
                !is_column_full
            })
            .collect();

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
    pub fn is_terminal(&self) -> Option<Value> {
        let all_pieces = self.p1_piece_board | self.p2_piece_board;

        if self.has_connected_4() {
            return Some(if self.p1_turn_to_move {
                Value([0.0, 1.0])
            } else {
                Value([1.0, 0.0])
            });
        }

        if all_pieces & TOP_ROW_MASK == TOP_ROW_MASK {
            return Some(Value([0.5, 0.5]));
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

    pub fn get_transposition_hash(&self) -> u64 {
        self.zobrist.board_state_hash()
    }

    fn has_connected_4(&self) -> bool {
        let board = if self.p1_turn_to_move {
            self.p2_piece_board
        } else {
            self.p1_piece_board
        };

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

        false
    }
}

impl Display for GameState {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let p1_board = map_board_to_arr(self.p1_piece_board);
        let p2_board = map_board_to_arr(self.p2_piece_board);

        writeln!(f)?;
        writeln!(f, "   +---+---+---+---+---+---+---+")?;

        for y in 0..6 {
            for x in 0..7 {
                if x == 0 {
                    write!(f, "   |")?;
                }
                let idx = y * 7 + x;
                let p = if p1_board[idx] != 0.0 {
                    "X"
                } else if p2_board[idx] != 0.0 {
                    "O"
                } else {
                    " "
                };
                write!(f, " {} |", p)?;
            }
            writeln!(f)?;
            if y != 5 {
                writeln!(f, "   |---+---+---+---+---+---+---|")?;
            }
        }

        writeln!(f, "   +---+---+---+---+---+---+---+")?;
        writeln!(f, "     1   2   3   4   5   6   7  ")?;

        Ok(())
    }
}

impl TranspositionHash for GameState {
    fn transposition_hash(&self) -> u64 {
        self.get_transposition_hash()
    }
}
