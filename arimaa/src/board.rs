use super::constants::{BOARD_SIZE, INPUT_C};
use half::f16;

pub fn set_board_bits_invertable(arr: &mut [f16], channel_idx: usize, board: u64, invert: bool) {
    let mut board = board;

    while board != 0 {
        let bit_idx = board.trailing_zeros() as usize;
        let removed_bit_board_idx = if invert { invert_idx(bit_idx) } else { bit_idx };

        let cell_idx = removed_bit_board_idx * INPUT_C + channel_idx;

        arr[cell_idx] = f16::ONE;

        board ^= 1 << bit_idx;
    }
}

fn invert_idx(idx: usize) -> usize {
    BOARD_SIZE - idx - 1
}

#[cfg(test)]
mod tests {
    use super::super::constants::INPUT_SIZE;
    use super::*;
    use arimaa_engine::{GameState, Piece, Square};
    use common::bits::single_bit_index_u64;

    fn square_to_idx(square: Square) -> usize {
        let bit_board = square.as_bit_board();
        single_bit_index_u64(bit_board)
    }

    fn value_at_square(vec: &[f16], offset: usize, col: char, row: usize) -> f16 {
        let idx = square_to_idx(Square::new(col, row)) * INPUT_C + offset;
        vec[idx]
    }

    fn num_values_set(vec: &[f16]) -> usize {
        vec.iter().filter(|v| **v != f16::ZERO).count()
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
            .parse()
            .unwrap();

        let mut result: Vec<f16> = Vec::with_capacity(INPUT_SIZE);
        result.extend(std::iter::repeat(f16::ZERO).take(INPUT_SIZE));
        let rabbit_offset = 5;
        set_board_bits_invertable(
            &mut result,
            rabbit_offset,
            game_state
                .get_piece_board()
                .get_bits_for_piece(Piece::Rabbit, true),
            false,
        );

        assert_eq!(num_values_set(&result), 1);
        assert_eq!(value_at_square(&result, rabbit_offset, 'a', 1), f16::ONE);

        // Inverted
        let mut result: Vec<f16> = Vec::with_capacity(INPUT_SIZE);
        result.extend(std::iter::repeat(f16::ZERO).take(INPUT_SIZE));
        set_board_bits_invertable(
            &mut result,
            rabbit_offset,
            game_state
                .get_piece_board()
                .get_bits_for_piece(Piece::Rabbit, true),
            true,
        );

        assert_eq!(num_values_set(&result), 1);
        assert_eq!(value_at_square(&result, rabbit_offset, 'h', 8), f16::ONE);
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
            .parse()
            .unwrap();

        let mut result: Vec<f16> = Vec::with_capacity(INPUT_SIZE);
        result.extend(std::iter::repeat(f16::ZERO).take(INPUT_SIZE));
        let rabbit_offset = 5;
        set_board_bits_invertable(
            &mut result,
            rabbit_offset,
            game_state
                .get_piece_board()
                .get_bits_for_piece(Piece::Rabbit, true),
            false,
        );

        assert_eq!(num_values_set(&result), 1);
        assert_eq!(value_at_square(&result, rabbit_offset, 'b', 8), f16::ONE);

        // Inverted
        let mut result: Vec<f16> = Vec::with_capacity(INPUT_SIZE);
        result.extend(std::iter::repeat(f16::ZERO).take(INPUT_SIZE));
        set_board_bits_invertable(
            &mut result,
            rabbit_offset,
            game_state
                .get_piece_board()
                .get_bits_for_piece(Piece::Rabbit, true),
            true,
        );

        assert_eq!(num_values_set(&result), 1);
        assert_eq!(value_at_square(&result, rabbit_offset, 'g', 1), f16::ONE);
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
            .parse()
            .unwrap();

        let mut result: Vec<f16> = Vec::with_capacity(INPUT_SIZE);
        result.extend(std::iter::repeat(f16::ZERO).take(INPUT_SIZE));
        let rabbit_offset = 5;
        set_board_bits_invertable(
            &mut result,
            rabbit_offset,
            game_state
                .get_piece_board()
                .get_bits_for_piece(Piece::Rabbit, true),
            false,
        );

        assert_eq!(num_values_set(&result), 1);
        assert_eq!(value_at_square(&result, rabbit_offset, 'd', 4), f16::ONE);

        // Inverted
        let mut result: Vec<f16> = Vec::with_capacity(INPUT_SIZE);
        result.extend(std::iter::repeat(f16::ZERO).take(INPUT_SIZE));
        set_board_bits_invertable(
            &mut result,
            rabbit_offset,
            game_state
                .get_piece_board()
                .get_bits_for_piece(Piece::Rabbit, true),
            true,
        );

        assert_eq!(num_values_set(&result), 1);
        assert_eq!(value_at_square(&result, rabbit_offset, 'e', 5), f16::ONE);
    }
}
