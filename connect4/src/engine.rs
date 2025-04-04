use engine::GameEngine;

use super::{Action, GameState, Predictions};

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
    type Terminal = Predictions;

    fn take_action(&self, game_state: &Self::State, action: &Self::Action) -> Self::State {
        match action {
            Action::DropPiece(column) => game_state.drop_piece(*column as usize),
        }
    }

    fn terminal_state(&self, game_state: &Self::State) -> Option<Self::Terminal> {
        game_state.is_terminal().map(|v| Predictions::new(v, 0.0))
    }

    fn player_to_move(&self, game_state: &Self::State) -> usize {
        if game_state.p1_turn_to_move {
            1
        } else {
            2
        }
    }

    fn move_number(&self, game_state: &Self::State) -> usize {
        game_state.p2_piece_board.count_ones() as usize + 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::GameState as GameStateTrait;

    #[test]
    fn test_new_state_is_correct() {
        let state: GameState = GameState::initial();
        assert!(state.p1_turn_to_move);
        assert_eq!(state.p1_piece_board, 0);
        assert_eq!(state.p2_piece_board, 0);
    }

    #[test]
    fn test_drop_piece_switches_player() {
        let mut state: GameState = GameState::initial();
        state = state.drop_piece(1);
        assert!(!state.p1_turn_to_move);
        state = state.drop_piece(1);
        assert!(state.p1_turn_to_move);
    }

    #[test]
    fn test_drop_piece_empty_first_column() {
        let mut state: GameState = GameState::initial();
        state = state.drop_piece(1);
        assert_eq!(state.p1_piece_board, 1);
    }

    #[test]
    fn test_drop_piece_empty_last_column() {
        let mut state: GameState = GameState::initial();
        state = state.drop_piece(7);
        assert_eq!(state.p1_piece_board, 1 << (7 * 6));
    }

    #[test]
    fn test_drop_piece_empty_column_includes_other_pieces() {
        let mut state: GameState = GameState::initial();
        state = state.drop_piece(1);
        state = state.drop_piece(2);
        state = state.drop_piece(3);

        let piece_1 = 1;
        let piece_3 = 1 << (7 * 2);
        assert_eq!(state.p1_piece_board, piece_1 | piece_3);
    }

    #[test]
    fn test_drop_piece_column_on_other_player_piece() {
        let mut state: GameState = GameState::initial();
        state = state.drop_piece(1);
        state = state.drop_piece(1);
        state = state.drop_piece(4);
        state = state.drop_piece(4);
        state = state.drop_piece(4);
        state = state.drop_piece(4);
        state = state.drop_piece(4);

        let piece_1_1 = 1;
        let piece_1_2 = 2;

        let column_4 = 1 << (7 * 3);
        let piece_4_1 = column_4;
        let piece_4_2 = column_4 << 1;
        let piece_4_3 = column_4 << 2;
        let piece_4_4 = column_4 << 3;
        let piece_4_5 = column_4 << 4;

        assert_eq!(
            state.p1_piece_board,
            piece_1_1 | piece_4_1 | piece_4_3 | piece_4_5
        );
        assert_eq!(state.p2_piece_board, piece_1_2 | piece_4_2 | piece_4_4);
    }

    #[test]
    fn test_get_valid_actions_all_columns_available() {
        let mut state: GameState = GameState::initial();

        for column in 1..8 {
            for _ in 1..6 {
                state = state.drop_piece(column);
            }
        }

        assert_eq!(
            state.get_valid_actions().as_slice(),
            [true, true, true, true, true, true, true]
        );
    }

    #[test]
    fn test_get_valid_actions_all_columns_full() {
        let mut state: GameState = GameState::initial();

        for column in 1..8 {
            for _ in 1..7 {
                state = state.drop_piece(column);
            }
        }

        assert_eq!(
            state.get_valid_actions().as_slice(),
            [false, false, false, false, false, false, false]
        );
    }

    #[test]
    fn test_get_valid_actions_first_column_full() {
        let mut state: GameState = GameState::initial();

        for column in 1..8 {
            for _ in 1..6 {
                state = state.drop_piece(column);
            }
        }

        state = state.drop_piece(1);

        assert_eq!(
            state.get_valid_actions().as_slice(),
            [false, true, true, true, true, true, true]
        );
    }

    #[test]
    fn test_get_valid_actions_last_column_full() {
        let mut state: GameState = GameState::initial();

        for column in 1..8 {
            for _ in 1..6 {
                state = state.drop_piece(column);
            }
        }

        state = state.drop_piece(7);

        assert_eq!(
            state.get_valid_actions().as_slice(),
            [true, true, true, true, true, true, false]
        );
    }

    #[test]
    fn test_get_valid_actions_three_columns_full() {
        let mut state: GameState = GameState::initial();

        for column in 1..8 {
            for _ in 1..6 {
                state = state.drop_piece(column);
            }
        }

        state = state.drop_piece(3);
        state = state.drop_piece(4);
        state = state.drop_piece(5);

        assert_eq!(
            state.get_valid_actions().as_slice(),
            [true, true, false, false, false, true, true]
        );
    }

    #[test]
    fn test_number_of_actions_none() {
        let state: GameState = GameState::initial();

        assert_eq!(state.number_of_actions(), 0);
    }

    #[test]
    fn test_number_of_actions_one() {
        let mut state: GameState = GameState::initial();
        state = state.drop_piece(3);

        assert_eq!(state.number_of_actions(), 1);
    }

    #[test]
    fn test_number_of_actions_two() {
        let mut state: GameState = GameState::initial();
        state = state.drop_piece(3);
        state = state.drop_piece(3);

        assert_eq!(state.number_of_actions(), 2);
    }

    #[test]
    fn test_number_of_actions_three() {
        let mut state: GameState = GameState::initial();
        state = state.drop_piece(3);
        state = state.drop_piece(3);
        state = state.drop_piece(4);

        assert_eq!(state.number_of_actions(), 3);
    }
}
