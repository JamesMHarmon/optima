use super::constants::{BOARD_WIDTH,BOARD_HEIGHT};
use common::bits::single_bit_index_u64;

pub fn map_board_to_arr_invertable(board: u64, invert: bool) -> Vec<f32> {
    let size = BOARD_HEIGHT * BOARD_WIDTH;
    let mut board = board;
    let mut result: Vec<f32> = Vec::with_capacity(size);
    result.extend(std::iter::repeat(0.0).take(size));

    while board != 0 {
        let board_without_first_bit = board & (board - 1);
        let removed_bit = board ^ board_without_first_bit;
        let removed_bit_idx = single_bit_index_u64(removed_bit);
        let removed_bit_vec_idx = if invert { invert_idx(removed_bit_idx) } else { removed_bit_idx };

        result[removed_bit_vec_idx] = 1.0;

        board = board_without_first_bit;
    }

    result
}

fn invert_idx(idx: usize) -> usize {
    (BOARD_WIDTH * BOARD_HEIGHT) - idx - 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::action::{Action,Direction,Square};
    use super::super::engine::GameState;
    use engine::game_state::GameState as GameStateTrait;

    fn square_to_idx(square: Square) -> usize {
        let bit_board = square.as_bit_board();
        let bit_index = single_bit_index_u64(bit_board);
        bit_index
    }

    fn value_at_square(vec: &[f32], col: char, row: usize) -> f32 {
        let idx = square_to_idx(Square::new(col, row));
        vec[idx]
    }

    fn num_values_set(vec: &[f32]) -> usize {
        vec.iter().filter(|v| **v != 0.0).count()
    }

    #[test]
    fn test_map_board_to_arr_invertable_pawn_i9() {
        let game_state = GameState::initial();
        let game_state = game_state.take_action(&Action::Move(Square::new('a', 2), Direction::Up));

        let arr = map_board_to_arr_invertable(
            game_state.p1_pawn_board,
            BoardType::Pawn,
            false
        );

        assert_eq!(num_values_set(&arr), 1);
        assert_eq!(value_at_square(&arr, 'i', 9), 1.0);

        // Inverted
        let arr = map_board_to_arr_invertable(
            game_state.p1_pawn_board,
            BoardType::Pawn,
            true
        );

        assert_eq!(num_values_set(&arr), 1);
        assert_eq!(value_at_square(&arr, 'a', 1), 1.0);
    }

    #[test]
    fn test_map_board_to_arr_invertable_pawn_a9() {
        let game_state = GameState::initial();
        let game_state = game_state.take_action(&Action::MovePawn(Square::new('a', 9)));

        let arr = map_board_to_arr_invertable(
            game_state.p1_pawn_board,
            BoardType::Pawn,
            false
        );

        assert_eq!(num_values_set(&arr), 1);
        assert_eq!(value_at_square(&arr, 'a', 9), 1.0);

        // Inverted
        let arr = map_board_to_arr_invertable(
            game_state.p1_pawn_board,
            BoardType::Pawn,
            true
        );

        assert_eq!(num_values_set(&arr), 1);
        assert_eq!(value_at_square(&arr, 'i', 1), 1.0);
    }

    #[test]
    fn test_map_board_to_arr_invertable_pawn_a1() {
        let game_state = GameState::initial();
        let game_state = game_state.take_action(&Action::MovePawn(Square::new('a', 1)));

        let arr = map_board_to_arr_invertable(
            game_state.p1_pawn_board,
            BoardType::Pawn,
            false
        );

        assert_eq!(num_values_set(&arr), 1);
        assert_eq!(value_at_square(&arr, 'a', 1), 1.0);

        // Inverted
        let arr = map_board_to_arr_invertable(
            game_state.p1_pawn_board,
            BoardType::Pawn,
            true
        );

        assert_eq!(num_values_set(&arr), 1);
        assert_eq!(value_at_square(&arr, 'i', 9), 1.0);
    }

    #[test]
    fn test_map_board_to_arr_invertable_pawn_h1() {
        let game_state = GameState::initial();
        let game_state = game_state.take_action(&Action::MovePawn(Square::new('i', 1)));

        let arr = map_board_to_arr_invertable(
            game_state.p1_pawn_board,
            BoardType::Pawn,
            false
        );

        assert_eq!(num_values_set(&arr), 1);
        assert_eq!(value_at_square(&arr, 'i', 1), 1.0);

        // Inverted
        let arr = map_board_to_arr_invertable(
            game_state.p1_pawn_board,
            BoardType::Pawn,
            true
        );

        assert_eq!(num_values_set(&arr), 1);
        assert_eq!(value_at_square(&arr, 'a', 9), 1.0);
    }

    #[test]
    fn test_map_board_to_arr_invertable_pawn_e5() {
        let game_state = GameState::initial();
        let game_state = game_state.take_action(&Action::MovePawn(Square::new('e', 5)));

        let arr = map_board_to_arr_invertable(
            game_state.p1_pawn_board,
            BoardType::Pawn,
            false
        );

        assert_eq!(num_values_set(&arr), 1);
        assert_eq!(value_at_square(&arr, 'e', 5), 1.0);

        // Inverted
        let arr = map_board_to_arr_invertable(
            game_state.p1_pawn_board,
            BoardType::Pawn,
            true
        );

        assert_eq!(num_values_set(&arr), 1);
        assert_eq!(value_at_square(&arr, 'e', 5), 1.0);
    }
}
