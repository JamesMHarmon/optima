#[derive(Debug)]
pub struct GameState {
    pub p1_turn_to_move: bool,
    pub p1_piece_board: u64,
    pub p2_piece_board: u64
}

impl GameState {
    pub fn new() -> Self {
        GameState {
            p1_turn_to_move: true,
            p1_piece_board: 0,
            p2_piece_board: 0
        }
    }

    pub fn drop_piece(&mut self, column: usize) {
        let column_adder = 1 << 7 * (column - 1);
        let all_pieces = self.p1_piece_board | self.p2_piece_board;
        let dropped_piece = (all_pieces + column_adder) & !all_pieces;
        self.p1_turn_to_move = !self.p1_turn_to_move;

        if !self.p1_turn_to_move {
            self.p1_piece_board = self.p1_piece_board | dropped_piece;
        } else {
            self.p2_piece_board = self.p2_piece_board | dropped_piece;
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_state_is_correct() {
        let state = GameState::new();
        assert_eq!(state.p1_turn_to_move, true);
        assert_eq!(state.p1_piece_board, 0);
        assert_eq!(state.p2_piece_board, 0);
    }

    #[test]
    fn test_drop_piece_switches_player() {
        let mut state = GameState::new();
        state.drop_piece(1);
        assert_eq!(state.p1_turn_to_move, false);
        state.drop_piece(1);
        assert_eq!(state.p1_turn_to_move, true);
    }

    #[test]
    fn test_drop_piece_empty_first_column() {
        let mut state = GameState::new();
        state.drop_piece(1);
        assert_eq!(state.p1_piece_board, 1);
    }

    #[test]
    fn test_drop_piece_empty_last_column() {
        let mut state = GameState::new();
        state.drop_piece(7);
        assert_eq!(state.p1_piece_board, 1 << 7 * 6);
    }

    #[test]
    fn test_drop_piece_empty_column_includes_other_pieces() {
        let mut state = GameState::new();
        state.drop_piece(1);
        state.drop_piece(2);
        state.drop_piece(3);

        let piece_1 = 1;
        let piece_3 = 1 << 7 * 2;
        assert_eq!(state.p1_piece_board, piece_1 | piece_3);
    }

    #[test]
    fn test_drop_piece_column_on_other_player_piece() {
        let mut state = GameState::new();
        state.drop_piece(1);
        state.drop_piece(1);
        state.drop_piece(4);
        state.drop_piece(4);
        state.drop_piece(4);
        state.drop_piece(4);
        state.drop_piece(4);

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
        let mut state = GameState::new();

        for column in 1..8 {
            for _ in 1..6 {
                state.drop_piece(column);
            }
        }

        assert_eq!(state.get_valid_actions().as_slice(), [true, true, true, true, true, true, true]);
    }

    #[test]
    fn test_get_valid_actions_all_columns_full() {
        let mut state = GameState::new();

        for column in 1..8 {
            for _ in 1..7 {
                state.drop_piece(column);
            }
        }

        assert_eq!(state.get_valid_actions().as_slice(), [false, false, false, false, false, false, false]);
    }

    #[test]
    fn test_get_valid_actions_first_column_full() {
        let mut state = GameState::new();

        for column in 1..8 {
            for _ in 1..6 {
                state.drop_piece(column);
            }
        }

        state.drop_piece(1);

        assert_eq!(state.get_valid_actions().as_slice(), [false, true, true, true, true, true, true]);
    }

    #[test]
    fn test_get_valid_actions_last_column_full() {
        let mut state = GameState::new();

        for column in 1..8 {
            for _ in 1..6 {
                state.drop_piece(column);
            }
        }

        state.drop_piece(7);

        assert_eq!(state.get_valid_actions().as_slice(), [true, true, true, true, true, true, false]);
    }

    #[test]
    fn test_get_valid_actions_three_columns_full() {
        let mut state = GameState::new();

        for column in 1..8 {
            for _ in 1..6 {
                state.drop_piece(column);
            }
        }

        state.drop_piece(3);
        state.drop_piece(4);
        state.drop_piece(5);

        assert_eq!(state.get_valid_actions().as_slice(), [true, true, false, false, false, true, true]);
    }
}

