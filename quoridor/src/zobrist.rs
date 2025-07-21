use super::zobrist_values::*;
use super::GameState;
use crate::constants::BOARD_SIZE;
use crate::constants::PAWN_BOARD_SIZE;
use crate::Coordinate;

#[derive(Hash, Eq, PartialEq, Clone, Copy, Debug)]
pub struct Zobrist {
    hash: u64,
}

impl Zobrist {
    pub fn initial() -> Self {
        Zobrist { hash: INITIAL }
    }

    #[must_use]
    pub fn move_pawn(&self, prev_game_state: &GameState, new_pawn_board: u128) -> Self {
        let move_piece_value = get_move_pawn_value(prev_game_state, new_pawn_board);

        let hash = self.hash ^ move_piece_value;

        Zobrist { hash }
    }

    #[must_use]
    pub fn place_wall(
        &self,
        prev_game_state: &GameState,
        place_wall_bit: u128,
        is_vertical: bool,
    ) -> Self {
        let wall_value = get_wall_value(
            Coordinate::from_bit_board(place_wall_bit).index(),
            is_vertical,
        );
        let walls_remaining_value = get_num_walls_placed_value(prev_game_state);

        let hash = self.hash ^ wall_value ^ walls_remaining_value;

        Zobrist { hash }
    }

    #[must_use]
    pub fn toggle_player_turn(&self) -> Self {
        let hash = self.hash ^ PLAYER_TO_MOVE;

        Zobrist { hash }
    }

    #[must_use]
    pub fn board_state_hash(&self) -> u64 {
        self.hash
    }

    /* Represents the initial hash position with negated state. */
    fn inverse_initial() -> u64 {
        INITIAL
            ^ get_pawn_hash_value("e1".parse::<Coordinate>().unwrap().index(), true)
            ^ get_pawn_hash_value("e9".parse::<Coordinate>().unwrap().index(), false)
            ^ WALLS_PLACED[10]
    }
}

impl From<&GameState> for Zobrist {
    fn from(game_state: &GameState) -> Self {
        let player_to_move_hash = if game_state.p1_turn_to_move() {
            0
        } else {
            PLAYER_TO_MOVE
        };

        let hash = Zobrist::inverse_initial()
            ^ player_to_move_hash
            ^ get_pawn_hash_value(game_state.player_info(1).pawn().index(), true)
            ^ get_pawn_hash_value(game_state.player_info(2).pawn().index(), false)
            ^ game_state
                .vertical_walls()
                .fold(0, |acc, coord| acc ^ get_wall_value(coord.index(), true))
            ^ game_state
                .horizontal_walls()
                .fold(0, |acc, coord| acc ^ get_wall_value(coord.index(), false))
            ^ WALLS_PLACED[game_state.player_info(1).num_walls()];

        Zobrist { hash }
    }
}

fn get_move_pawn_value(prev_game_state: &GameState, new_pawn_board: u128) -> u64 {
    let is_p1 = prev_game_state.p1_turn_to_move();
    let source_coord = prev_game_state.curr_player().pawn();
    let dest_coord = Coordinate::from_bit_board(new_pawn_board);

    let source_coord_value = get_pawn_hash_value(source_coord.index(), is_p1);
    let dest_coord_value = get_pawn_hash_value(dest_coord.index(), is_p1);

    source_coord_value ^ dest_coord_value
}

fn get_pawn_hash_value(coord_index: usize, is_p1: bool) -> u64 {
    let pawn_offset = if is_p1 { 0 } else { PAWN_BOARD_SIZE };

    PAWN[pawn_offset + coord_index]
}

fn get_wall_value(coord_index: usize, is_vertical: bool) -> u64 {
    let wall_type_offset = if is_vertical { 0 } else { BOARD_SIZE };

    WALL[wall_type_offset + coord_index]
}

fn get_num_walls_placed_value(prev_game_state: &GameState) -> u64 {
    // This hash only needs to track the remaining walls for p1.
    // Placing the wall itself will change the hash.
    if !prev_game_state.p1_turn_to_move() {
        return 0;
    }

    let prev_num_walls_remaining = prev_game_state.player_info(1).num_walls();

    let walls_placed_prev_value = WALLS_PLACED[prev_num_walls_remaining];
    let walls_placed_new_value = WALLS_PLACED[prev_num_walls_remaining - 1];

    walls_placed_prev_value ^ walls_placed_new_value
}

#[cfg(test)]
mod tests {
    use crate::Zobrist;

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

    #[test]
    fn test_hash_from_game_state() {
        let mut game_state = GameState::initial();
        let zobrist: Zobrist = (&game_state).into();
        assert_eq!(game_state.transposition_hash(), zobrist.board_state_hash());

        let actions = ["e2", "e8", "e3", "e7", "a2h", "e6", "a4v", "a6h", "a8h"]
            .iter()
            .map(|s| s.parse::<Action>().unwrap());

        for action in actions {
            game_state.take_action(&action);
            let zobrist: Zobrist = (&game_state).into();
            assert_eq!(game_state.transposition_hash(), zobrist.board_state_hash());
        }
    }
}
