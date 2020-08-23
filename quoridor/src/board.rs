use super::constants::{BOARD_WIDTH,BOARD_HEIGHT};

#[derive(PartialEq)]
pub enum BoardType {
    Pawn,
    VerticalWall,
    HorizontalWall
}

#[allow(clippy::assign_op_pattern)]
pub fn map_board_to_arr_invertable(board: u128, board_type: BoardType, invert: bool) -> Vec<f32> {
    let size = BOARD_HEIGHT * BOARD_WIDTH;
    let mut board = board;
    let mut result: Vec<f32> = Vec::with_capacity(size);
    result.extend(std::iter::repeat(0.0).take(size));

    if invert && (board_type == BoardType::VerticalWall || board_type == BoardType::HorizontalWall) {
        // Shift the walls up and to the right so that when we do a 180 rotation, they will be in their respective positions.
        board = shift_up_right!(board);
    }

    while board != 0 {
        let bit_idx = board.trailing_zeros() as usize;
        let removed_bit_vec_idx = if invert { bit_idx } else { map_board_idx_to_vec_idx(bit_idx) };

        result[removed_bit_vec_idx] = 1.0;

        // Walls cover two squares, we want both of the covered locations to be part of the output.
        if board_type == BoardType::VerticalWall {
            result[removed_bit_vec_idx - BOARD_WIDTH] = 1.0;
        }

        if board_type == BoardType::HorizontalWall {
            result[removed_bit_vec_idx + 1] = 1.0;
        }

        board ^= 1 << bit_idx;
    }

    result
}

/// Converts from the bit_board index, which starts with the least significant bit at the bottom right and traverses right to left.
/// From:
/// 00 01 02 03 04 05 06 07 08
/// 09 10 11 12 13 14 15 16 17
/// 18 19 20 21 22 23 24 25 26
/// 27 28 29 30 31 32 33 34 35
/// 36 37 38 39 40 41 42 43 44
/// 45 46 47 48 49 50 51 52 53
/// 54 55 56 57 58 59 60 61 62
/// 63 64 65 66 67 68 69 70 71
/// 72 73 74 75 76 77 78 79 80
///
/// To the vector form which starts in the top left and goes left to right.
/// To:
/// 80 79 78 77 76 75 74 73 72
/// 71 70 69 68 67 66 65 64 63
/// 62 61 60 59 58 57 56 55 54
/// 53 52 51 50 49 48 47 46 45
/// 44 43 42 41 40 39 38 37 36
/// 35 34 33 32 31 30 29 28 27
/// 26 25 24 23 22 21 20 19 18
/// 17 16 15 14 13 12 11 10 09
/// 08 07 06 05 04 03 02 01 00
fn map_board_idx_to_vec_idx(board_idx: usize) -> usize {
    (BOARD_WIDTH * BOARD_HEIGHT) - board_idx - 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::action::{Action,Coordinate};
    use super::super::engine::GameState;
    use engine::game_state::GameState as GameStateTrait;
    use common::bits::single_bit_index;

    fn coordinate_to_idx(coord: Coordinate) -> usize {
        let bit_board = coord.as_bit_board();
        let single_bit_index = single_bit_index(bit_board);
        let vec_idx = map_board_idx_to_vec_idx(single_bit_index);
        vec_idx
    }

    fn value_at_coordinate(vec: &[f32], col: char, row: usize) -> f32 {
        let idx = coordinate_to_idx(Coordinate::new(col, row));
        vec[idx]
    }

    fn num_values_set(vec: &[f32]) -> usize {
        vec.iter().filter(|v| **v != 0.0).count()
    }

    #[test]
    fn test_map_board_idx_to_vec_idx_bottom_left() {
        let board_idx = 72;
        let expected_vec_idx = 08;
        let actual_vec_index = map_board_idx_to_vec_idx(board_idx);
        assert_eq!(expected_vec_idx, actual_vec_index);
    }

    #[test]
    fn test_map_board_idx_to_vec_idx_top_left() {
        let board_idx = 00;
        let expected_vec_idx = 80;
        let actual_vec_index = map_board_idx_to_vec_idx(board_idx);
        assert_eq!(expected_vec_idx, actual_vec_index);
    }

    #[test]
    fn test_map_board_idx_to_vec_idx_bottom_right() {
        let board_idx = 80;
        let expected_vec_idx = 00;
        let actual_vec_index = map_board_idx_to_vec_idx(board_idx);
        assert_eq!(expected_vec_idx, actual_vec_index);
    }

    #[test]
    fn test_map_board_idx_to_vec_idx_top_right() {
        let board_idx = 72;
        let expected_vec_idx = 08;
        let actual_vec_index = map_board_idx_to_vec_idx(board_idx);
        assert_eq!(expected_vec_idx, actual_vec_index);
    }

    #[test]
    fn test_map_board_to_arr_invertable_horizontal_walls_h8() {
        let game_state = GameState::initial();
        let game_state = game_state.take_action(&Action::PlaceHorizontalWall(Coordinate::new('h', 8)));

        let arr = map_board_to_arr_invertable(
            game_state.horizontal_wall_placement_board,
            BoardType::HorizontalWall,
            false
        );

        assert_eq!(num_values_set(&arr), 2);
        assert_eq!(value_at_coordinate(&arr, 'h', 8), 1.0);
        assert_eq!(value_at_coordinate(&arr, 'i', 8), 1.0);

        // Inverted
        let arr = map_board_to_arr_invertable(
            game_state.horizontal_wall_placement_board,
            BoardType::HorizontalWall,
            true
        );

        assert_eq!(num_values_set(&arr), 2);
        assert_eq!(value_at_coordinate(&arr, 'a', 1), 1.0);
        assert_eq!(value_at_coordinate(&arr, 'b', 1), 1.0);
    }

    #[test]
    fn test_map_board_to_arr_invertable_horizontal_walls_a8() {
        let game_state = GameState::initial();
        let game_state = game_state.take_action(&Action::PlaceHorizontalWall(Coordinate::new('a', 8)));

        let arr = map_board_to_arr_invertable(
            game_state.horizontal_wall_placement_board,
            BoardType::HorizontalWall,
            false
        );

        assert_eq!(num_values_set(&arr), 2);
        assert_eq!(value_at_coordinate(&arr, 'a', 8), 1.0);
        assert_eq!(value_at_coordinate(&arr, 'b', 8), 1.0);

        // Inverted
        let arr = map_board_to_arr_invertable(
            game_state.horizontal_wall_placement_board,
            BoardType::HorizontalWall,
            true
        );

        assert_eq!(num_values_set(&arr), 2);
        assert_eq!(value_at_coordinate(&arr, 'h', 1), 1.0);
        assert_eq!(value_at_coordinate(&arr, 'i', 1), 1.0);
    }

    #[test]
    fn test_map_board_to_arr_invertable_horizontal_walls_a1() {
        let game_state = GameState::initial();
        let game_state = game_state.take_action(&Action::PlaceHorizontalWall(Coordinate::new('a', 1)));

        let arr = map_board_to_arr_invertable(
            game_state.horizontal_wall_placement_board,
            BoardType::HorizontalWall,
            false
        );

        assert_eq!(num_values_set(&arr), 2);
        assert_eq!(value_at_coordinate(&arr, 'a', 1), 1.0);
        assert_eq!(value_at_coordinate(&arr, 'b', 1), 1.0);

        // Inverted
        let arr = map_board_to_arr_invertable(
            game_state.horizontal_wall_placement_board,
            BoardType::HorizontalWall,
            true
        );

        assert_eq!(num_values_set(&arr), 2);
        assert_eq!(value_at_coordinate(&arr, 'h', 8), 1.0);
        assert_eq!(value_at_coordinate(&arr, 'i', 8), 1.0);
    }

    #[test]
    fn test_map_board_to_arr_invertable_horizontal_walls_h1() {
        let game_state = GameState::initial();
        let game_state = game_state.take_action(&Action::PlaceHorizontalWall(Coordinate::new('h', 1)));

        let arr = map_board_to_arr_invertable(
            game_state.horizontal_wall_placement_board,
            BoardType::HorizontalWall,
            false
        );

        assert_eq!(num_values_set(&arr), 2);
        assert_eq!(value_at_coordinate(&arr, 'h', 1), 1.0);
        assert_eq!(value_at_coordinate(&arr, 'i', 1), 1.0);

        // Inverted
        let arr = map_board_to_arr_invertable(
            game_state.horizontal_wall_placement_board,
            BoardType::HorizontalWall,
            true
        );

        assert_eq!(num_values_set(&arr), 2);
        assert_eq!(value_at_coordinate(&arr, 'a', 8), 1.0);
        assert_eq!(value_at_coordinate(&arr, 'b', 8), 1.0);
    }

    #[test]
    fn test_map_board_to_arr_invertable_horizontal_walls_e5() {
        let game_state = GameState::initial();
        let game_state = game_state.take_action(&Action::PlaceHorizontalWall(Coordinate::new('e', 5)));

        let arr = map_board_to_arr_invertable(
            game_state.horizontal_wall_placement_board,
            BoardType::HorizontalWall,
            false
        );

        assert_eq!(num_values_set(&arr), 2);
        assert_eq!(value_at_coordinate(&arr, 'e', 5), 1.0);
        assert_eq!(value_at_coordinate(&arr, 'f', 5), 1.0);

        // Inverted
        let arr = map_board_to_arr_invertable(
            game_state.horizontal_wall_placement_board,
            BoardType::HorizontalWall,
            true
        );

        assert_eq!(num_values_set(&arr), 2);
        assert_eq!(value_at_coordinate(&arr, 'd', 4), 1.0);
        assert_eq!(value_at_coordinate(&arr, 'e', 4), 1.0);
    }

    #[test]
    fn test_map_board_to_arr_invertable_vertical_walls_h8() {
        let game_state = GameState::initial();
        let game_state = game_state.take_action(&Action::PlaceVerticalWall(Coordinate::new('h', 8)));

        let arr = map_board_to_arr_invertable(
            game_state.vertical_wall_placement_board,
            BoardType::VerticalWall,
            false
        );

        assert_eq!(num_values_set(&arr), 2);
        assert_eq!(value_at_coordinate(&arr, 'h', 8), 1.0);
        assert_eq!(value_at_coordinate(&arr, 'h', 9), 1.0);

        // Inverted
        let arr = map_board_to_arr_invertable(
            game_state.vertical_wall_placement_board,
            BoardType::VerticalWall,
            true
        );

        assert_eq!(num_values_set(&arr), 2);
        assert_eq!(value_at_coordinate(&arr, 'a', 1), 1.0);
        assert_eq!(value_at_coordinate(&arr, 'a', 2), 1.0);
    }

    #[test]
    fn test_map_board_to_arr_invertable_vertical_walls_a8() {
        let game_state = GameState::initial();
        let game_state = game_state.take_action(&Action::PlaceVerticalWall(Coordinate::new('a', 8)));

        let arr = map_board_to_arr_invertable(
            game_state.vertical_wall_placement_board,
            BoardType::VerticalWall,
            false
        );

        assert_eq!(num_values_set(&arr), 2);
        assert_eq!(value_at_coordinate(&arr, 'a', 8), 1.0);
        assert_eq!(value_at_coordinate(&arr, 'a', 9), 1.0);

        // Inverted
        let arr = map_board_to_arr_invertable(
            game_state.vertical_wall_placement_board,
            BoardType::VerticalWall,
            true
        );

        assert_eq!(num_values_set(&arr), 2);
        assert_eq!(value_at_coordinate(&arr, 'h', 1), 1.0);
        assert_eq!(value_at_coordinate(&arr, 'h', 2), 1.0);
    }

    #[test]
    fn test_map_board_to_arr_invertable_vertical_walls_a1() {
        let game_state = GameState::initial();
        let game_state = game_state.take_action(&Action::PlaceVerticalWall(Coordinate::new('a', 1)));

        let arr = map_board_to_arr_invertable(
            game_state.vertical_wall_placement_board,
            BoardType::VerticalWall,
            false
        );

        assert_eq!(num_values_set(&arr), 2);
        assert_eq!(value_at_coordinate(&arr, 'a', 1), 1.0);
        assert_eq!(value_at_coordinate(&arr, 'a', 2), 1.0);

        // Inverted
        let arr = map_board_to_arr_invertable(
            game_state.vertical_wall_placement_board,
            BoardType::VerticalWall,
            true
        );

        assert_eq!(num_values_set(&arr), 2);
        assert_eq!(value_at_coordinate(&arr, 'h', 8), 1.0);
        assert_eq!(value_at_coordinate(&arr, 'h', 9), 1.0);
    }

    #[test]
    fn test_map_board_to_arr_invertable_vertical_walls_h1() {
        let game_state = GameState::initial();
        let game_state = game_state.take_action(&Action::PlaceVerticalWall(Coordinate::new('h', 1)));

        let arr = map_board_to_arr_invertable(
            game_state.vertical_wall_placement_board,
            BoardType::VerticalWall,
            false
        );

        assert_eq!(num_values_set(&arr), 2);
        assert_eq!(value_at_coordinate(&arr, 'h', 1), 1.0);
        assert_eq!(value_at_coordinate(&arr, 'h', 2), 1.0);

        // Inverted
        let arr = map_board_to_arr_invertable(
            game_state.vertical_wall_placement_board,
            BoardType::VerticalWall,
            true
        );

        assert_eq!(num_values_set(&arr), 2);
        assert_eq!(value_at_coordinate(&arr, 'a', 8), 1.0);
        assert_eq!(value_at_coordinate(&arr, 'a', 9), 1.0);
    }

    #[test]
    fn test_map_board_to_arr_invertable_vertical_walls_e5() {
        let game_state = GameState::initial();
        let game_state = game_state.take_action(&Action::PlaceVerticalWall(Coordinate::new('e', 5)));

        let arr = map_board_to_arr_invertable(
            game_state.vertical_wall_placement_board,
            BoardType::VerticalWall,
            false
        );

        assert_eq!(num_values_set(&arr), 2);
        assert_eq!(value_at_coordinate(&arr, 'e', 5), 1.0);
        assert_eq!(value_at_coordinate(&arr, 'e', 6), 1.0);

        // Inverted
        let arr = map_board_to_arr_invertable(
            game_state.vertical_wall_placement_board,
            BoardType::VerticalWall,
            true
        );

        assert_eq!(num_values_set(&arr), 2);
        assert_eq!(value_at_coordinate(&arr, 'd', 4), 1.0);
        assert_eq!(value_at_coordinate(&arr, 'd', 5), 1.0);
    }


    #[test]
    fn test_map_board_to_arr_invertable_pawn_i9() {
        let game_state = GameState::initial();
        let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('i', 9)));

        let arr = map_board_to_arr_invertable(
            game_state.p1_pawn_board,
            BoardType::Pawn,
            false
        );

        assert_eq!(num_values_set(&arr), 1);
        assert_eq!(value_at_coordinate(&arr, 'i', 9), 1.0);

        // Inverted
        let arr = map_board_to_arr_invertable(
            game_state.p1_pawn_board,
            BoardType::Pawn,
            true
        );

        assert_eq!(num_values_set(&arr), 1);
        assert_eq!(value_at_coordinate(&arr, 'a', 1), 1.0);
    }

    #[test]
    fn test_map_board_to_arr_invertable_pawn_a9() {
        let game_state = GameState::initial();
        let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('a', 9)));

        let arr = map_board_to_arr_invertable(
            game_state.p1_pawn_board,
            BoardType::Pawn,
            false
        );

        assert_eq!(num_values_set(&arr), 1);
        assert_eq!(value_at_coordinate(&arr, 'a', 9), 1.0);

        // Inverted
        let arr = map_board_to_arr_invertable(
            game_state.p1_pawn_board,
            BoardType::Pawn,
            true
        );

        assert_eq!(num_values_set(&arr), 1);
        assert_eq!(value_at_coordinate(&arr, 'i', 1), 1.0);
    }

    #[test]
    fn test_map_board_to_arr_invertable_pawn_a1() {
        let game_state = GameState::initial();
        let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('a', 1)));

        let arr = map_board_to_arr_invertable(
            game_state.p1_pawn_board,
            BoardType::Pawn,
            false
        );

        assert_eq!(num_values_set(&arr), 1);
        assert_eq!(value_at_coordinate(&arr, 'a', 1), 1.0);

        // Inverted
        let arr = map_board_to_arr_invertable(
            game_state.p1_pawn_board,
            BoardType::Pawn,
            true
        );

        assert_eq!(num_values_set(&arr), 1);
        assert_eq!(value_at_coordinate(&arr, 'i', 9), 1.0);
    }

    #[test]
    fn test_map_board_to_arr_invertable_pawn_h1() {
        let game_state = GameState::initial();
        let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('i', 1)));

        let arr = map_board_to_arr_invertable(
            game_state.p1_pawn_board,
            BoardType::Pawn,
            false
        );

        assert_eq!(num_values_set(&arr), 1);
        assert_eq!(value_at_coordinate(&arr, 'i', 1), 1.0);

        // Inverted
        let arr = map_board_to_arr_invertable(
            game_state.p1_pawn_board,
            BoardType::Pawn,
            true
        );

        assert_eq!(num_values_set(&arr), 1);
        assert_eq!(value_at_coordinate(&arr, 'a', 9), 1.0);
    }

    #[test]
    fn test_map_board_to_arr_invertable_pawn_e5() {
        let game_state = GameState::initial();
        let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('e', 5)));

        let arr = map_board_to_arr_invertable(
            game_state.p1_pawn_board,
            BoardType::Pawn,
            false
        );

        assert_eq!(num_values_set(&arr), 1);
        assert_eq!(value_at_coordinate(&arr, 'e', 5), 1.0);

        // Inverted
        let arr = map_board_to_arr_invertable(
            game_state.p1_pawn_board,
            BoardType::Pawn,
            true
        );

        assert_eq!(num_values_set(&arr), 1);
        assert_eq!(value_at_coordinate(&arr, 'e', 5), 1.0);
    }
}
