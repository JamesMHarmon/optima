use super::constants::{BOARD_WIDTH,BOARD_HEIGHT};
use common::bits::single_bit_index_u64;

pub fn set_board_bits_invertable(arr: &mut [f32], board: u64, invert: bool) {
    let mut board = board;

    while board != 0 {
        let board_without_first_bit = board & (board - 1);
        let removed_bit = board ^ board_without_first_bit;
        let removed_bit_idx = single_bit_index_u64(removed_bit);
        let removed_bit_vec_idx = if invert { invert_idx(removed_bit_idx) } else { removed_bit_idx };

        arr[removed_bit_vec_idx] = 1.0;

        board = board_without_first_bit;
    }
}

pub fn set_placement_board_bits(arr: &mut [f32], board: u64) {
    let mut board = board;

    while board != 0 {
        let board_without_first_bit = board & (board - 1);
        let removed_bit = board ^ board_without_first_bit;
        let removed_bit_idx = single_bit_index_u64(removed_bit);
        let removed_bit_vec_idx = if removed_bit_idx < 16 { removed_bit_idx } else { removed_bit_idx - 32 };

        arr[removed_bit_vec_idx] = 1.0;

        board = board_without_first_bit;
    }
}

fn invert_idx(idx: usize) -> usize {
    (BOARD_WIDTH * BOARD_HEIGHT) - idx - 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::action::{Piece,Square};
    use super::super::engine::GameState;

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
    fn test_map_board_to_arr_invertable_pawn_a1() {
        let game_state: GameState = "
             1g
              +-----------------+
             8|                 |
             7|                 |
             6|     x     x     |
             5|                 |
             4|                 |
             3|     x     x     |
             2|                 |
             1| R               |
              +-----------------+
                a b c d e f g h"
            .parse().unwrap();

        let mut result: Vec<f32> = Vec::with_capacity(BOARD_WIDTH * BOARD_HEIGHT);
        result.extend(std::iter::repeat(0.0).take(BOARD_WIDTH * BOARD_HEIGHT));
        set_board_bits_invertable(
            &mut result,
            game_state.get_piece_board_for_step(game_state.get_current_step()).get_bits_for_piece(&Piece::Rabbit, true),
            false
        );

        assert_eq!(num_values_set(&result), 1);
        assert_eq!(value_at_square(&result, 'a', 1), 1.0);

        // Inverted
        let mut result: Vec<f32> = Vec::with_capacity(BOARD_WIDTH * BOARD_HEIGHT);
        result.extend(std::iter::repeat(0.0).take(BOARD_WIDTH * BOARD_HEIGHT));
        set_board_bits_invertable(
            &mut result,
            game_state.get_piece_board_for_step(game_state.get_current_step()).get_bits_for_piece(&Piece::Rabbit, true),
            true
        );

        assert_eq!(num_values_set(&result), 1);
        assert_eq!(value_at_square(&result, 'h', 8), 1.0);
    }

    #[test]
    fn test_set_board_bits_invertable_pawn_b8() {
        let game_state: GameState = "
             1g
              +-----------------+
             8|   R             |
             7|                 |
             6|     x     x     |
             5|                 |
             4|                 |
             3|     x     x     |
             2|                 |
             1|                 |
              +-----------------+
                a b c d e f g h"
            .parse().unwrap();

        let mut result: Vec<f32> = Vec::with_capacity(BOARD_WIDTH * BOARD_HEIGHT);
        result.extend(std::iter::repeat(0.0).take(BOARD_WIDTH * BOARD_HEIGHT));
        set_board_bits_invertable(
            &mut result,
            game_state.get_piece_board_for_step(game_state.get_current_step()).get_bits_for_piece(&Piece::Rabbit, true),
            false
        );

        assert_eq!(num_values_set(&result), 1);
        assert_eq!(value_at_square(&result, 'b', 8), 1.0);

        // Inverted
        let mut result: Vec<f32> = Vec::with_capacity(BOARD_WIDTH * BOARD_HEIGHT);
        result.extend(std::iter::repeat(0.0).take(BOARD_WIDTH * BOARD_HEIGHT));
        set_board_bits_invertable(
            &mut result,
            game_state.get_piece_board_for_step(game_state.get_current_step()).get_bits_for_piece(&Piece::Rabbit, true),
            true
        );

        assert_eq!(num_values_set(&result), 1);
        assert_eq!(value_at_square(&result, 'g', 1), 1.0);
    }

    #[test]
    fn test_set_board_bits_invertable_pawn_d4() {
        let game_state: GameState = "
             1g
              +-----------------+
             8|                 |
             7|                 |
             6|     x     x     |
             5|                 |
             4|       R         |
             3|     x     x     |
             2|                 |
             1|                 |
              +-----------------+
                a b c d e f g h"
            .parse().unwrap();

        let mut result: Vec<f32> = Vec::with_capacity(BOARD_WIDTH * BOARD_HEIGHT);
        result.extend(std::iter::repeat(0.0).take(BOARD_WIDTH * BOARD_HEIGHT));
        set_board_bits_invertable(
            &mut result,
            game_state.get_piece_board_for_step(game_state.get_current_step()).get_bits_for_piece(&Piece::Rabbit, true),
            false
        );

        assert_eq!(num_values_set(&result), 1);
        assert_eq!(value_at_square(&result, 'd', 4), 1.0);

        // Inverted
        let mut result: Vec<f32> = Vec::with_capacity(BOARD_WIDTH * BOARD_HEIGHT);
        result.extend(std::iter::repeat(0.0).take(BOARD_WIDTH * BOARD_HEIGHT));
        set_board_bits_invertable(
            &mut result,
            game_state.get_piece_board_for_step(game_state.get_current_step()).get_bits_for_piece(&Piece::Rabbit, true),
            true
        );

        assert_eq!(num_values_set(&result), 1);
        assert_eq!(value_at_square(&result, 'e', 5), 1.0);
    }
}
