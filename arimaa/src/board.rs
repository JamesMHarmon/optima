use super::constants::{BOARD_SIZE,PLAY_INPUT_C,PLACE_INPUT_C};

pub fn set_board_bits_invertable(arr: &mut [f32], offset: usize, board: u64, invert: bool) {
    let mut board = board;

    while board != 0 {
        let bit_idx = board.trailing_zeros() as usize;
        let removed_bit_board_idx = if invert { invert_idx(bit_idx) } else { bit_idx };

        let cell_idx = removed_bit_board_idx * PLAY_INPUT_C + offset;

        arr[cell_idx] = 1.0;

        board = board ^ 1 << bit_idx;
    }
}

pub fn set_placement_board_bits(arr: &mut [f32], offset: usize, board: u64) {
    let mut board = board;

    while board != 0 {
        let bit_idx = board.trailing_zeros() as usize;
        let removed_bit_board_idx = if bit_idx < 16 { bit_idx } else { bit_idx - 32 };

        let cell_idx = removed_bit_board_idx * PLACE_INPUT_C + offset;

        arr[cell_idx] = 1.0;

        board = board ^ 1 << bit_idx;
    }
}

fn invert_idx(idx: usize) -> usize {
    BOARD_SIZE - idx - 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::constants::{PLAY_INPUT_SIZE};
    use super::super::action::{Piece,Square};
    use super::super::engine::GameState;
    use common::bits::single_bit_index_u64;

    fn square_to_idx(square: Square) -> usize {
        let bit_board = square.as_bit_board();
        let bit_index = single_bit_index_u64(bit_board);
        bit_index
    }

    fn value_at_square(vec: &[f32], offset: usize, col: char, row: usize) -> f32 {
        let idx = square_to_idx(Square::new(col, row)) * PLAY_INPUT_C + offset;
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

        let mut result: Vec<f32> = Vec::with_capacity(PLAY_INPUT_SIZE);
        result.extend(std::iter::repeat(0.0).take(PLAY_INPUT_SIZE));
        let rabbit_offset = 5;
        set_board_bits_invertable(
            &mut result,
            rabbit_offset,
            game_state.get_piece_board_for_step(game_state.get_current_step()).get_bits_for_piece(Piece::Rabbit, true),
            false
        );

        assert_eq!(num_values_set(&result), 1);
        assert_eq!(value_at_square(&result, rabbit_offset, 'a', 1), 1.0);

        // Inverted
        let mut result: Vec<f32> = Vec::with_capacity(PLAY_INPUT_SIZE);
        result.extend(std::iter::repeat(0.0).take(PLAY_INPUT_SIZE));
        set_board_bits_invertable(
            &mut result,
            rabbit_offset,
            game_state.get_piece_board_for_step(game_state.get_current_step()).get_bits_for_piece(Piece::Rabbit, true),
            true
        );

        assert_eq!(num_values_set(&result), 1);
        assert_eq!(value_at_square(&result, rabbit_offset, 'h', 8), 1.0);
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

        let mut result: Vec<f32> = Vec::with_capacity(PLAY_INPUT_SIZE);
        result.extend(std::iter::repeat(0.0).take(PLAY_INPUT_SIZE));
        let rabbit_offset = 5;
        set_board_bits_invertable(
            &mut result,
            rabbit_offset,
            game_state.get_piece_board_for_step(game_state.get_current_step()).get_bits_for_piece(Piece::Rabbit, true),
            false
        );

        assert_eq!(num_values_set(&result), 1);
        assert_eq!(value_at_square(&result, rabbit_offset, 'b', 8), 1.0);

        // Inverted
        let mut result: Vec<f32> = Vec::with_capacity(PLAY_INPUT_SIZE);
        result.extend(std::iter::repeat(0.0).take(PLAY_INPUT_SIZE));
        set_board_bits_invertable(
            &mut result,
            rabbit_offset,
            game_state.get_piece_board_for_step(game_state.get_current_step()).get_bits_for_piece(Piece::Rabbit, true),
            true
        );

        assert_eq!(num_values_set(&result), 1);
        assert_eq!(value_at_square(&result, rabbit_offset, 'g', 1), 1.0);
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

        let mut result: Vec<f32> = Vec::with_capacity(PLAY_INPUT_SIZE);
        result.extend(std::iter::repeat(0.0).take(PLAY_INPUT_SIZE));
        let rabbit_offset = 5;
        set_board_bits_invertable(
            &mut result,
            rabbit_offset,
            game_state.get_piece_board_for_step(game_state.get_current_step()).get_bits_for_piece(Piece::Rabbit, true),
            false
        );

        assert_eq!(num_values_set(&result), 1);
        assert_eq!(value_at_square(&result, rabbit_offset, 'd', 4), 1.0);

        // Inverted
        let mut result: Vec<f32> = Vec::with_capacity(PLAY_INPUT_SIZE);
        result.extend(std::iter::repeat(0.0).take(PLAY_INPUT_SIZE));
        set_board_bits_invertable(
            &mut result,
            rabbit_offset,
            game_state.get_piece_board_for_step(game_state.get_current_step()).get_bits_for_piece(Piece::Rabbit, true),
            true
        );

        assert_eq!(num_values_set(&result), 1);
        assert_eq!(value_at_square(&result, rabbit_offset, 'e', 5), 1.0);
    }
}
