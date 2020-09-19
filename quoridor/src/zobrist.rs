use common::bits::single_bit_index;

use super::engine::GameState;
use super::zobrist_values::*;
use crate::constants::BOARD_SIZE;
use crate::constants::NUM_WALLS_PER_PLAYER;
use crate::constants::PAWN_BOARD_SIZE;

#[derive(Hash, Eq, PartialEq, Clone, Copy, Debug)]
pub struct Zobrist {
    hash: u64,
}

impl Zobrist {
    pub fn initial() -> Self {
        Zobrist { hash: INITIAL }
    }

    pub fn move_pawn(&self, prev_game_state: &GameState, pawn_board: u128) -> Self {
        let move_piece_value = get_move_pawn_value(prev_game_state, pawn_board);

        let hash = self.hash ^ PLAYER_TO_MOVE ^ move_piece_value;

        Zobrist { hash }
    }

    pub fn place_wall(
        &self,
        prev_game_state: &GameState,
        place_wall_bit: u128,
        is_vertical: bool,
    ) -> Self {
        let wall_value = get_wall_value(place_wall_bit, is_vertical);
        let walls_remaining_value = get_num_walls_placed_value(prev_game_state);

        let hash = self.hash ^ PLAYER_TO_MOVE ^ wall_value ^ walls_remaining_value;

        Zobrist { hash }
    }

    pub fn board_state_hash(&self) -> u64 {
        self.hash
    }
}

fn get_move_pawn_value(prev_game_state: &GameState, source_pawn_board: u128) -> u64 {
    let is_p1_turn_to_move = prev_game_state.p1_turn_to_move;
    let source_coord_bit = if is_p1_turn_to_move {
        prev_game_state.p1_pawn_board
    } else {
        prev_game_state.p2_pawn_board
    };
    let pawn_offset = if is_p1_turn_to_move {
        0
    } else {
        PAWN_BOARD_SIZE
    };

    let source_coord_value = PAWN[pawn_offset + single_bit_index(source_coord_bit)];
    let dest_coord_value = PAWN[pawn_offset + single_bit_index(source_pawn_board)];

    source_coord_value ^ dest_coord_value
}

fn get_wall_value(place_wall_bit: u128, is_vertical: bool) -> u64 {
    let wall_type_offset = if is_vertical { 0 } else { BOARD_SIZE };

    WALL[wall_type_offset + single_bit_index(place_wall_bit)]
}

fn get_num_walls_placed_value(prev_game_state: &GameState) -> u64 {
    let is_p1_turn_to_move = prev_game_state.p1_turn_to_move;
    let (num_walls_placed, walls_placed_offset) = if is_p1_turn_to_move {
        (prev_game_state.p1_num_walls_placed, 0)
    } else {
        (
            prev_game_state.p2_num_walls_placed,
            NUM_WALLS_PER_PLAYER + 1,
        )
    };

    let walls_placed_prev_value = WALLS_PLACED[num_walls_placed + walls_placed_offset];
    let walls_placed_new_value = WALLS_PLACED[num_walls_placed + 1 + walls_placed_offset];

    walls_placed_prev_value ^ walls_placed_new_value
}
