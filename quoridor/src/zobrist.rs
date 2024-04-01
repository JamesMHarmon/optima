use common::bits::single_bit_index;

use super::zobrist_values::*;
use super::GameState;
use crate::constants::BOARD_SIZE;
use crate::constants::PAWN_BOARD_SIZE;

#[derive(Hash, Eq, PartialEq, Clone, Copy, Debug)]
pub struct Zobrist {
    hash: u64,
}

impl Zobrist {
    pub fn initial() -> Self {
        Zobrist { hash: INITIAL }
    }

    pub fn move_pawn(&self, prev_game_state: &GameState, new_pawn_board: u128) -> Self {
        let move_piece_value = get_move_pawn_value(prev_game_state, new_pawn_board);

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

fn get_move_pawn_value(prev_game_state: &GameState, new_pawn_board: u128) -> u64 {
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
    let dest_coord_value = PAWN[pawn_offset + single_bit_index(new_pawn_board)];

    source_coord_value ^ dest_coord_value
}

fn get_wall_value(place_wall_bit: u128, is_vertical: bool) -> u64 {
    let wall_type_offset = if is_vertical { 0 } else { BOARD_SIZE };

    WALL[wall_type_offset + single_bit_index(place_wall_bit)]
}

fn get_num_walls_placed_value(prev_game_state: &GameState) -> u64 {
    // This hash only needs to track the remaining walls for p1.
    // Placing the wall itself will change the hash.
    if !prev_game_state.p1_turn_to_move {
        return 0;
    }

    let prev_num_p1_walls_placed = prev_game_state.p1_num_walls_placed as usize;

    let walls_placed_prev_value = WALLS_PLACED[prev_num_p1_walls_placed];
    let walls_placed_new_value = WALLS_PLACED[prev_num_p1_walls_placed + 1];

    walls_placed_prev_value ^ walls_placed_new_value
}

#[cfg(test)]
mod tests {
    use super::super::{Action, GameState};
    use engine::game_state::GameState as GameStateTrait;

    /*
       Tests that if player 1 and player 2 move their pawns and end up back in the same spots, then transposition is equivalent.
    */
    #[test]
    fn test_transposition_hash_pawn_moves() {
        let mut game_state = GameState::initial();
        let initial_hash = game_state.transposition_hash();

        game_state.take_action(&"f1".parse::<Action>().unwrap());
        game_state.take_action(&"f9".parse::<Action>().unwrap());
        game_state.take_action(&"e1".parse::<Action>().unwrap());
        game_state.take_action(&"e9".parse::<Action>().unwrap());

        let after_actions_hash = game_state.transposition_hash();

        assert_eq!(initial_hash, after_actions_hash);
    }

    /*
       Tests that if player 1 and player 2 both place a wall, it doesn't depend on which player placed which wall.
    */
    #[test]
    fn test_transposition_hash_alternate_player_wall_placement() {
        let mut game_state = GameState::initial();

        game_state.take_action(&"f4h".parse::<Action>().unwrap());
        game_state.take_action(&"f6h".parse::<Action>().unwrap());

        let after_actions_hash = game_state.transposition_hash();

        let mut game_state = GameState::initial();

        game_state.take_action(&"f6h".parse::<Action>().unwrap());
        game_state.take_action(&"f4h".parse::<Action>().unwrap());

        let after_actions_hash_2 = game_state.transposition_hash();

        assert_eq!(after_actions_hash, after_actions_hash_2);
    }

    /*
       Tests that the transposition is not equivalent when player 1 places the wall in contrast to player 2 placing the walls.
       These are not the same as player 1 will have less walls remaining to place than player 2.
    */
    #[test]
    fn test_transposition_hash_who_made_wall_placement() {
        let mut game_state = GameState::initial();

        game_state.take_action(&"f1".parse::<Action>().unwrap());
        game_state.take_action(&"f4h".parse::<Action>().unwrap());
        game_state.take_action(&"e1".parse::<Action>().unwrap());
        game_state.take_action(&"f6h".parse::<Action>().unwrap());

        let after_actions_hash = game_state.transposition_hash();

        let mut game_state = GameState::initial();

        game_state.take_action(&"f4h".parse::<Action>().unwrap());
        game_state.take_action(&"f9".parse::<Action>().unwrap());
        game_state.take_action(&"f6h".parse::<Action>().unwrap());
        game_state.take_action(&"e9".parse::<Action>().unwrap());

        let after_actions_hash_2 = game_state.transposition_hash();

        assert_ne!(after_actions_hash, after_actions_hash_2);
    }
}
