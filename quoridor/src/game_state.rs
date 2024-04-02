use crate::ActionType;

use super::constants::{MAX_NUMBER_OF_MOVES, NUM_WALLS_PER_PLAYER};
use super::{Action, Coordinate, Value, Zobrist};
use engine::game_state;

const LEFT_COLUMN_MASK: u128 =                                  0b__100000000__100000000__100000000__100000000__100000000__100000000__100000000__100000000__100000000;
const RIGHT_COLUMN_MASK: u128 =                                 0b__000000001__000000001__000000001__000000001__000000001__000000001__000000001__000000001__000000001;

const VALID_PIECE_POSITION_MASK: u128 =                         0b__111111111__111111111__111111111__111111111__111111111__111111111__111111111__111111111__111111111;
const CANDIDATE_WALL_PLACEMENT_MASK: u128 =                     0b__000000000__111111110__111111110__111111110__111111110__111111110__111111110__111111110__111111110;
const P1_OBJECTIVE_MASK: u128 =                                 0b__111111111__000000000__000000000__000000000__000000000__000000000__000000000__000000000__000000000;
const P2_OBJECTIVE_MASK: u128 =                                 0b__000000000__000000000__000000000__000000000__000000000__000000000__000000000__000000000__111111111;
const P1_STARTING_POS_MASK: u128 =                              0b__000000000__000000000__000000000__000000000__000000000__000000000__000000000__000000000__000010000;
const P2_STARTING_POS_MASK: u128 =                              0b__000010000__000000000__000000000__000000000__000000000__000000000__000000000__000000000__000000000;

const HORIZONTAL_WALL_LEFT_EDGE_TOUCHING_BOARD_MASK: u128 =     0b__000000000__100000000__100000000__100000000__100000000__100000000__100000000__100000000__100000000;
const HORIZONTAL_WALL_RIGHT_EDGE_TOUCHING_BOARD_MASK: u128 =    0b__000000000__000000010__000000010__000000010__000000010__000000010__000000010__000000010__000000010;
const VERTICAL_WALL_TOP_EDGE_TOUCHING_BOARD_MASK: u128 =        0b__000000000__111111110__000000000__000000000__000000000__000000000__000000000__000000000__000000000;
const VERTICAL_WALL_BOTTOM_EDGE_TOUCHING_BOARD_MASK: u128 =     0b__000000000__000000000__000000000__000000000__000000000__000000000__000000000__000000000__111111110;

#[derive(Hash, Clone, Debug)]
pub struct GameState {
    pub move_number: usize,
    pub p1_turn_to_move: bool,
    pub p1_pawn_board: u128,
    pub p2_pawn_board: u128,
    pub p1_num_walls_placed: u8,
    pub p2_num_walls_placed: u8,
    pub vertical_wall_board: u128,
    pub horizontal_wall_board: u128,
    pub zobrist: Zobrist,
}

#[derive(Debug)]
struct PathingResult {
    has_path: bool,
    path: u128,
}

impl GameState {
    pub fn take_action(&mut self, action: &Action) {
        let coord = action.coord();

        match action.action_type() {
            ActionType::PawnMove => self.move_pawn(coord),
            ActionType::VerticalWall => self.place_wall(coord, true),
            ActionType::HorizontalWall => self.place_wall(coord, false),
        }

        self.increment_turn();
    }

    pub fn get_valid_pawn_move_actions(&self) -> impl Iterator<Item = Action> {
        Self::bit_board_coords_to_actions(self.valid_pawn_moves(), ActionType::PawnMove)
    }

    pub fn get_valid_horizontal_wall_actions(&self) -> impl Iterator<Item = Action> {
        Self::bit_board_coords_to_actions(
            self.valid_horizontal_wall_placement(),
            ActionType::HorizontalWall,
        )
    }

    pub fn get_valid_vertical_wall_actions(&self) -> impl Iterator<Item = Action> {
        Self::bit_board_coords_to_actions(
            self.valid_vertical_wall_placement(),
            ActionType::VerticalWall,
        )
    }

    pub fn bit_board_coords_to_actions(
        bit_board: u128,
        action_type: ActionType,
    ) -> impl Iterator<Item = Action> {
        Self::map_bit_board_to_coordinates(bit_board)
            .into_iter()
            .map(move |coord| Action::new(action_type, coord))
    }

    pub fn is_terminal(&self) -> Option<Value> {
        let pawn_board = if self.p1_turn_to_move {
            self.p2_pawn_board
        } else {
            self.p1_pawn_board
        };
        let objective_mask = if self.p1_turn_to_move {
            P2_OBJECTIVE_MASK
        } else {
            P1_OBJECTIVE_MASK
        };

        if pawn_board & objective_mask != 0 {
            Some(if self.p1_turn_to_move {
                Value([0.0, 1.0])
            } else {
                Value([1.0, 0.0])
            })
        } else if self.move_number >= MAX_NUMBER_OF_MOVES {
            // A game that runs too long will be a loss for both players.
            Some(Value([0.0, 0.0]))
        } else {
            None
        }
    }

    pub fn get_vertical_symmetry(&self) -> Self {
        let get_vertical_symmetry_bit_board = |bit_board: u128, shift: bool| {
            Self::map_bit_board_to_coordinates(bit_board)
                .into_iter()
                .fold(0u128, |mut bit_board, coord| {
                    bit_board |= coord.vertical_symmetry(shift).as_bit_board();
                    bit_board
                })
        };

        Self {
            p1_pawn_board: get_vertical_symmetry_bit_board(self.p1_pawn_board, false),
            p2_pawn_board: get_vertical_symmetry_bit_board(self.p2_pawn_board, false),
            vertical_wall_board: get_vertical_symmetry_bit_board(self.vertical_wall_board, true),
            horizontal_wall_board: get_vertical_symmetry_bit_board(
                self.horizontal_wall_board,
                true,
            ),
            move_number: self.move_number,
            p1_num_walls_placed: self.p1_num_walls_placed,
            p2_num_walls_placed: self.p2_num_walls_placed,
            p1_turn_to_move: self.p1_turn_to_move,
            // @TODO: Need to update the zobrist hash here.
            zobrist: self.zobrist,
        }
    }

    pub fn transposition_hash(&self) -> u64 {
        self.zobrist.board_state_hash()
    }

    pub fn player_to_move(&self) -> usize {
        if self.p1_turn_to_move {
            1
        } else {
            2
        }
    }

    fn move_pawn(&mut self, coord: Coordinate) {
        let pawn_board = coord.as_bit_board();
        self.zobrist = self.zobrist.move_pawn(self, pawn_board);

        if self.p1_turn_to_move {
            self.p1_pawn_board = pawn_board
        } else {
            self.p2_pawn_board = pawn_board
        }
    }

    fn place_wall(&mut self, coord: Coordinate, is_vertical: bool) {
        let wall_placement = coord.as_bit_board();
        self.zobrist = self.zobrist.place_wall(self, wall_placement, is_vertical);

        let p1_turn_to_move = self.p1_turn_to_move;
        if p1_turn_to_move {
            self.p1_num_walls_placed += 1;
        } else {
            self.p2_num_walls_placed += 1;
        }

        if is_vertical {
            self.vertical_wall_board |= wall_placement;
        } else {
            self.horizontal_wall_board |= wall_placement;
        }
    }

    fn increment_turn(&mut self) {
        self.p1_turn_to_move = !self.p1_turn_to_move;
        if self.p1_turn_to_move {
            self.move_number += 1;
        }
    }

    fn valid_pawn_moves(&self) -> u128 {
        let active_player_board = self.get_active_player_board();
        let opposing_player_board = self.get_opposing_player_board();

        let move_up_mask = self.get_move_up_mask();
        let move_right_mask = self.get_move_right_mask();
        let move_down_mask = self.get_move_down_mask();
        let move_left_mask = self.get_move_left_mask();

        let up_move = shift_up!(active_player_board) & move_up_mask;
        let right_move = shift_right!(active_player_board) & move_right_mask;
        let down_move = shift_down!(active_player_board) & move_down_mask;
        let left_move = shift_left!(active_player_board) & move_left_mask;

        let valid_moves: u128 = up_move | right_move | down_move | left_move;
        let overlapping_move: u128 = valid_moves & opposing_player_board;

        if overlapping_move == 0 {
            return valid_moves;
        }

        let overlap_up_move = up_move & opposing_player_board;
        let overlap_right_move = right_move & opposing_player_board;
        let overlap_down_move = down_move & opposing_player_board;
        let overlap_left_move = left_move & opposing_player_board;

        let straight_jump_up_move = shift_up!(overlap_up_move) & move_up_mask;
        let straight_jump_right_move = shift_right!(overlap_right_move) & move_right_mask;
        let straight_jump_down_move = shift_down!(overlap_down_move) & move_down_mask;
        let straight_jump_left_move = shift_left!(overlap_left_move) & move_left_mask;

        let straight_jump_move = straight_jump_up_move
            | straight_jump_right_move
            | straight_jump_down_move
            | straight_jump_left_move;

        if straight_jump_move != 0 {
            return valid_moves & !opposing_player_board | straight_jump_move;
        }

        let side_jump_moves = (shift_up!(overlapping_move) & move_up_mask
            | shift_right!(overlapping_move) & move_right_mask
            | shift_down!(overlapping_move) & move_down_mask
            | shift_left!(overlapping_move) & move_left_mask)
            & !active_player_board;

        valid_moves & !opposing_player_board | side_jump_moves
    }

    fn get_active_player_board(&self) -> u128 {
        if self.p1_turn_to_move {
            self.p1_pawn_board
        } else {
            self.p2_pawn_board
        }
    }

    fn get_opposing_player_board(&self) -> u128 {
        if self.p1_turn_to_move {
            self.p2_pawn_board
        } else {
            self.p1_pawn_board
        }
    }

    fn get_vertical_wall_blocks(&self) -> u128 {
        shift_up!(self.vertical_wall_board) | self.vertical_wall_board
    }

    fn get_horizontal_wall_blocks(&self) -> u128 {
        shift_right!(self.horizontal_wall_board) | self.horizontal_wall_board
    }

    fn valid_horizontal_wall_placement(&self) -> u128 {
        if self.active_player_has_wall_to_place() {
            self.get_valid_horizontal_wall_positions()
        } else {
            0
        }
    }

    fn valid_vertical_wall_placement(&self) -> u128 {
        if self.active_player_has_wall_to_place() {
            self.get_valid_vertical_wall_positions()
        } else {
            0
        }
    }

    fn active_player_has_wall_to_place(&self) -> bool {
        if self.p1_turn_to_move {
            self.p1_num_walls_placed < NUM_WALLS_PER_PLAYER
        } else {
            self.p2_num_walls_placed < NUM_WALLS_PER_PLAYER
        }
    }

    fn get_valid_horizontal_wall_positions(&self) -> u128 {
        let candidate_horizontal_wall_placement = self.get_candidate_horizontal_wall_placement();
        let invalid_horizontal_candidates = self.get_invalid_horizontal_wall_candidates();
        candidate_horizontal_wall_placement & !invalid_horizontal_candidates
    }

    fn get_valid_vertical_wall_positions(&self) -> u128 {
        let candidate_vertical_wall_placement = self.get_candidate_vertical_wall_placement();
        let invalid_vertical_candidates = self.get_invalid_vertical_wall_candidates();
        candidate_vertical_wall_placement & !invalid_vertical_candidates
    }

    fn get_candidate_horizontal_wall_placement(&self) -> u128 {
        !(self.horizontal_wall_board
            | shift_right!(self.horizontal_wall_board)
            | shift_left!(self.horizontal_wall_board))
            & !self.vertical_wall_board
            & CANDIDATE_WALL_PLACEMENT_MASK
    }

    fn get_candidate_vertical_wall_placement(&self) -> u128 {
        !(self.vertical_wall_board
            | shift_down!(self.vertical_wall_board)
            | shift_up!(self.vertical_wall_board))
            & !self.horizontal_wall_board
            & CANDIDATE_WALL_PLACEMENT_MASK
    }

    fn get_invalid_horizontal_wall_candidates(&self) -> u128 {
        let mut invalid_placements: u128 = 0;
        let mut horizontal_connecting_candidates = self.get_horizontal_connecting_candidates();

        while horizontal_connecting_candidates != 0 {
            let with_removed_candidate =
                horizontal_connecting_candidates & (horizontal_connecting_candidates - 1);
            let removed_candidate = with_removed_candidate ^ horizontal_connecting_candidates;

            let state_with_candidate = Self {
                horizontal_wall_board: self.horizontal_wall_board | removed_candidate,
                ..*self
            };

            if !state_with_candidate.players_have_path() {
                invalid_placements |= removed_candidate;
            }

            horizontal_connecting_candidates = with_removed_candidate;
        }

        invalid_placements
    }

    fn get_invalid_vertical_wall_candidates(&self) -> u128 {
        let mut invalid_placements: u128 = 0;
        let mut vertical_connecting_candidates = self.get_vertical_connecting_candidates();

        while vertical_connecting_candidates != 0 {
            let with_removed_candidate =
                vertical_connecting_candidates & (vertical_connecting_candidates - 1);
            let removed_candidate = with_removed_candidate ^ vertical_connecting_candidates;

            let state_with_candidate = Self {
                vertical_wall_board: self.vertical_wall_board | removed_candidate,
                ..*self
            };

            if !state_with_candidate.players_have_path() {
                invalid_placements |= removed_candidate;
            }

            vertical_connecting_candidates = with_removed_candidate;
        }

        invalid_placements
    }

    fn players_have_path(&self) -> bool {
        let p1_path_result = self.find_path(self.p1_pawn_board, P1_OBJECTIVE_MASK);

        if !p1_path_result.has_path {
            return false;
        }

        // If the pathing result generated from checking for p1's path overlaps the p2 pawn, we can start where that path left off to
        // save a few cycles.
        let p2_path_start = if p1_path_result.path & self.p2_pawn_board != 0 {
            p1_path_result.path
        } else {
            self.p2_pawn_board
        };

        self.find_path(p2_path_start, P2_OBJECTIVE_MASK).has_path
    }

    fn find_path(&self, start: u128, end: u128) -> PathingResult {
        let up_mask = self.get_move_up_mask();
        let right_mask = self.get_move_right_mask();
        let down_mask = self.get_move_down_mask();
        let left_mask = self.get_move_left_mask();

        let mut path = start;

        loop {
            // MOVE UP & DOWN & LEFT & RIGHT
            let up_path = shift_up!(path) & up_mask;
            let right_path = shift_right!(path) & right_mask;
            let down_path = shift_down!(path) & down_mask;
            let left_path = shift_left!(path) & left_mask;
            let updated_path = up_path | right_path | down_path | left_path | path;

            // Check if the objective is reachable
            if end & updated_path != 0 {
                return PathingResult {
                    has_path: true,
                    path: updated_path,
                };
            }

            // Check if any progress was made, if there is no progress from the last iteration then we are stuck.
            if updated_path == path {
                return PathingResult {
                    has_path: false,
                    path: updated_path,
                };
            }

            path = updated_path;
        }
    }

    fn get_move_up_mask(&self) -> u128 {
        !(shift_up!(self.get_horizontal_wall_blocks())) & VALID_PIECE_POSITION_MASK
    }

    fn get_move_right_mask(&self) -> u128 {
        !shift_right!(self.get_vertical_wall_blocks())
            & VALID_PIECE_POSITION_MASK
            & !LEFT_COLUMN_MASK
    }

    fn get_move_down_mask(&self) -> u128 {
        !self.get_horizontal_wall_blocks() & VALID_PIECE_POSITION_MASK
    }

    fn get_move_left_mask(&self) -> u128 {
        !self.get_vertical_wall_blocks() & VALID_PIECE_POSITION_MASK & !RIGHT_COLUMN_MASK
    }

    fn get_horizontal_connecting_candidates(&self) -> u128 {
        let candidate_horizontal_walls = self.get_candidate_horizontal_wall_placement();
        let horizontal_walls = self.horizontal_wall_board;
        let vertical_walls = self.vertical_wall_board;

        // Compare to existing walls. Wall segment candidates check against locations of possible new connections. These are checked in combinations since the walls may already be connected. We don't want to count the same existing connection twice or thrice.
        let middle_touching =
            candidate_horizontal_walls & (shift_up!(vertical_walls) | shift_down!(vertical_walls));
        let left_edge_touching = candidate_horizontal_walls
            & (shift_right!(vertical_walls)
                | shift_down_right!(vertical_walls)
                | shift_up_right!(vertical_walls)
                | shift_right!(shift_right!(horizontal_walls))
                | HORIZONTAL_WALL_LEFT_EDGE_TOUCHING_BOARD_MASK);
        let right_edge_touching = candidate_horizontal_walls
            & (shift_left!(vertical_walls)
                | shift_down_left!(vertical_walls)
                | shift_up_left!(vertical_walls)
                | shift_left!(shift_left!(horizontal_walls))
                | HORIZONTAL_WALL_RIGHT_EDGE_TOUCHING_BOARD_MASK);

        (left_edge_touching & middle_touching)
            | (left_edge_touching & right_edge_touching)
            | (middle_touching & right_edge_touching)
    }

    fn get_vertical_connecting_candidates(&self) -> u128 {
        let candidate_vertical_walls = self.get_candidate_vertical_wall_placement();
        let vertical_walls = self.vertical_wall_board;
        let horizontal_walls = self.horizontal_wall_board;

        // Compare to existing walls. Wall segment candidates check against locations of possible new connections. These are checked in combinations since the walls may already be connected. We don't want to count the same existing connection twice or thrice.
        let middle_touching = candidate_vertical_walls
            & (shift_right!(horizontal_walls) | shift_left!(horizontal_walls));
        let top_edge_touching = candidate_vertical_walls
            & (shift_down_left!(horizontal_walls)
                | shift_down!(horizontal_walls)
                | shift_down_right!(horizontal_walls)
                | shift_down!(shift_down!(vertical_walls))
                | VERTICAL_WALL_TOP_EDGE_TOUCHING_BOARD_MASK);
        let bottom_edge_touching = candidate_vertical_walls
            & (shift_up_left!(horizontal_walls)
                | shift_up!(horizontal_walls)
                | shift_up_right!(horizontal_walls)
                | shift_up!(shift_up!(vertical_walls))
                | VERTICAL_WALL_BOTTOM_EDGE_TOUCHING_BOARD_MASK);

        (top_edge_touching & middle_touching)
            | (top_edge_touching & bottom_edge_touching)
            | (middle_touching & bottom_edge_touching)
    }

    fn map_bit_board_to_coordinates(board: u128) -> Vec<Coordinate> {
        let mut board = board;
        let mut coordinates = Vec::with_capacity(board.count_ones() as usize);

        while board != 0 {
            let bit_idx = board.trailing_zeros();
            let removed_bit = 1 << bit_idx;
            let coordinate = Coordinate::from_bit_board(removed_bit);
            coordinates.push(coordinate);

            board ^= removed_bit;
        }

        coordinates
    }
}

impl game_state::GameState for GameState {
    fn initial() -> Self {
        GameState {
            move_number: 1,
            p1_turn_to_move: true,
            p1_pawn_board: P1_STARTING_POS_MASK,
            p2_pawn_board: P2_STARTING_POS_MASK,
            p1_num_walls_placed: 0,
            p2_num_walls_placed: 0,
            vertical_wall_board: 0,
            horizontal_wall_board: 0,
            zobrist: Zobrist::initial(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::action::Action;
    use super::super::value::Value;
    use super::GameState;
    use engine::game_state::GameState as GameStateTrait;

    fn intersects(actions: &[Action], exclusions: &[Action]) -> bool {
        actions.iter().any(|a| exclusions.iter().any(|a2| a == a2))
    }

    #[test]
    fn test_get_valid_pawn_move_actions_p1() {
        let game_state = GameState::initial();
        let valid_actions = game_state.get_valid_pawn_move_actions().collect::<Vec<_>>();

        assert_eq!(
            valid_actions,
            vec!(
                "f1".parse::<Action>().unwrap(),
                "d1".parse::<Action>().unwrap(),
                "e2".parse::<Action>().unwrap()
            )
        );
    }

    #[test]
    fn test_get_valid_pawn_move_actions_p2() {
        let mut game_state = GameState::initial();
        game_state.take_action(&"f1".parse::<Action>().unwrap());
        let valid_actions = game_state.get_valid_pawn_move_actions().collect::<Vec<_>>();

        assert_eq!(
            valid_actions,
            vec!(
                "e8".parse::<Action>().unwrap(),
                "f9".parse::<Action>().unwrap(),
                "d9".parse::<Action>().unwrap()
            )
        );
    }

    #[test]
    fn test_get_valid_pawn_move_actions_vertical_wall() {
        let mut game_state = GameState::initial();
        game_state.take_action(&"d1v".parse::<Action>().unwrap());
        game_state.take_action(&"e1v".parse::<Action>().unwrap());
        let valid_actions = game_state.get_valid_pawn_move_actions().collect::<Vec<_>>();

        assert_eq!(valid_actions, vec!("e2".parse::<Action>().unwrap()));
    }

    #[test]
    fn test_get_valid_pawn_move_actions_vertical_wall_top() {
        let mut game_state = GameState::initial();
        game_state.take_action(&"e2".parse::<Action>().unwrap());
        game_state.take_action(&"e7".parse::<Action>().unwrap());
        game_state.take_action(&"d1v".parse::<Action>().unwrap());
        game_state.take_action(&"e1v".parse::<Action>().unwrap());
        let valid_actions = game_state.get_valid_pawn_move_actions().collect::<Vec<_>>();

        assert_eq!(
            valid_actions,
            vec!(
                "e1".parse::<Action>().unwrap(),
                "e3".parse::<Action>().unwrap()
            )
        );
    }

    #[test]
    fn test_get_valid_pawn_move_actions_horizontal_wall() {
        let mut game_state = GameState::initial();
        game_state.take_action(&"d8h".parse::<Action>().unwrap());
        game_state.take_action(&"e1h".parse::<Action>().unwrap());
        let valid_actions = game_state.get_valid_pawn_move_actions().collect::<Vec<_>>();

        assert_eq!(
            valid_actions,
            vec!(
                "f1".parse::<Action>().unwrap(),
                "d1".parse::<Action>().unwrap()
            )
        );

        game_state.take_action(&"f1".parse::<Action>().unwrap());
        let valid_actions = game_state.get_valid_pawn_move_actions().collect::<Vec<_>>();

        assert_eq!(
            valid_actions,
            vec!(
                "f9".parse::<Action>().unwrap(),
                "d9".parse::<Action>().unwrap()
            )
        );

        game_state.take_action(&"f9".parse::<Action>().unwrap());
        let valid_actions = game_state.get_valid_pawn_move_actions().collect::<Vec<_>>();

        assert_eq!(
            valid_actions,
            vec!(
                "g1".parse::<Action>().unwrap(),
                "e1".parse::<Action>().unwrap()
            )
        );
    }

    #[test]
    fn test_get_valid_pawn_move_actions_blocked() {
        let mut game_state = GameState::initial();
        game_state.take_action(&"e2".parse::<Action>().unwrap());
        game_state.take_action(&"e8".parse::<Action>().unwrap());
        game_state.take_action(&"e3".parse::<Action>().unwrap());
        game_state.take_action(&"e7".parse::<Action>().unwrap());
        game_state.take_action(&"e4".parse::<Action>().unwrap());
        game_state.take_action(&"e6".parse::<Action>().unwrap());
        game_state.take_action(&"e5".parse::<Action>().unwrap());

        let valid_actions = game_state.get_valid_pawn_move_actions().collect::<Vec<_>>();
        assert_eq!(
            valid_actions,
            vec!(
                "e4".parse::<Action>().unwrap(),
                "f6".parse::<Action>().unwrap(),
                "d6".parse::<Action>().unwrap(),
                "e7".parse::<Action>().unwrap()
            )
        );

        game_state.take_action(&"e4h".parse::<Action>().unwrap());
        game_state.take_action(&"a1h".parse::<Action>().unwrap());
        let valid_actions = game_state.get_valid_pawn_move_actions().collect::<Vec<_>>();

        assert_eq!(
            valid_actions,
            vec!(
                "f5".parse::<Action>().unwrap(),
                "d5".parse::<Action>().unwrap(),
                "f6".parse::<Action>().unwrap(),
                "d6".parse::<Action>().unwrap(),
                "e7".parse::<Action>().unwrap()
            )
        );

        game_state.take_action(&"e6h".parse::<Action>().unwrap());
        let valid_actions = game_state.get_valid_pawn_move_actions().collect::<Vec<_>>();

        assert_eq!(
            valid_actions,
            vec!(
                "f5".parse::<Action>().unwrap(),
                "d5".parse::<Action>().unwrap(),
                "f6".parse::<Action>().unwrap(),
                "d6".parse::<Action>().unwrap()
            )
        );
    }

    #[test]
    fn test_get_valid_horizontal_wall_actions_initial() {
        let game_state = GameState::initial();
        let valid_actions = game_state
            .get_valid_horizontal_wall_actions()
            .collect::<Vec<_>>();

        let mut cols = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];
        let rows = [1, 2, 3, 4, 5, 6, 7, 8];
        cols.reverse();

        let mut actions = Vec::new();

        for row in rows.iter() {
            for col in cols.iter() {
                actions.push(format!("{}{}h", col, row).parse().unwrap());
            }
        }

        assert_eq!(valid_actions.len(), actions.len());
        assert_eq!(valid_actions, actions);
    }

    #[test]
    fn test_get_valid_horizontal_wall_actions_on_horizontal_wall() {
        let mut game_state = GameState::initial();
        game_state.take_action(&"d1h".parse::<Action>().unwrap());

        let valid_actions = game_state
            .get_valid_horizontal_wall_actions()
            .collect::<Vec<_>>();
        let excludes_actions = vec![
            "c1h".parse::<Action>().unwrap(),
            "d1h".parse::<Action>().unwrap(),
            "e1h".parse::<Action>().unwrap(),
        ];
        let intersects = intersects(&valid_actions, &excludes_actions);

        assert!(!intersects);
        assert_eq!(valid_actions.len(), 64 - excludes_actions.len());
    }

    #[test]
    fn test_get_valid_horizontal_wall_actions_on_vertical_wall() {
        let mut game_state = GameState::initial();
        game_state.take_action(&"e5v".parse::<Action>().unwrap());

        let valid_actions = game_state
            .get_valid_horizontal_wall_actions()
            .collect::<Vec<_>>();
        let excludes_actions = vec!["e5h".parse::<Action>().unwrap()];
        let intersects = intersects(&valid_actions, &excludes_actions);

        assert!(!intersects);
        assert_eq!(valid_actions.len(), 64 - excludes_actions.len());
    }

    #[test]
    fn test_get_valid_horizontal_wall_actions_blocking_path() {
        let mut game_state = GameState::initial();
        game_state.take_action(&"c1v".parse::<Action>().unwrap());
        game_state.take_action(&"e1v".parse::<Action>().unwrap());

        let valid_actions = game_state
            .get_valid_horizontal_wall_actions()
            .collect::<Vec<_>>();
        let excludes_actions = vec![
            "c1h".parse::<Action>().unwrap(),
            "e1h".parse::<Action>().unwrap(),
            "d1h".parse::<Action>().unwrap(),
            "d2h".parse::<Action>().unwrap(),
        ];
        let intersects = intersects(&valid_actions, &excludes_actions);

        assert!(!intersects);
        assert_eq!(valid_actions.len(), 64 - excludes_actions.len());
    }

    #[test]
    fn test_get_valid_horizontal_wall_actions_blocking_path_other_player() {
        let mut game_state = GameState::initial();
        game_state.take_action(&"c1v".parse::<Action>().unwrap());
        game_state.take_action(&"e1v".parse::<Action>().unwrap());
        game_state.take_action(&"e2".parse::<Action>().unwrap());

        let valid_actions = game_state
            .get_valid_horizontal_wall_actions()
            .collect::<Vec<_>>();
        let excludes_actions = vec![
            "c1h".parse::<Action>().unwrap(),
            "e1h".parse::<Action>().unwrap(),
            "d2h".parse::<Action>().unwrap(),
        ];
        let intersects = intersects(&valid_actions, &excludes_actions);

        assert!(!intersects);
        assert_eq!(valid_actions.len(), 64 - excludes_actions.len());
    }

    #[test]
    fn test_get_valid_horizontal_wall_actions_blocking_path_vert_horz() {
        let mut game_state = GameState::initial();
        game_state.take_action(&"c1v".parse::<Action>().unwrap());
        game_state.take_action(&"e1v".parse::<Action>().unwrap());
        game_state.take_action(&"c2h".parse::<Action>().unwrap());

        let valid_actions = game_state
            .get_valid_horizontal_wall_actions()
            .collect::<Vec<_>>();
        let excludes_actions = vec![
            "c1h".parse::<Action>().unwrap(),
            "e1h".parse::<Action>().unwrap(),
            "d2h".parse::<Action>().unwrap(),
            "b2h".parse::<Action>().unwrap(),
            "c2h".parse::<Action>().unwrap(),
            "d2h".parse::<Action>().unwrap(),
            "e2h".parse::<Action>().unwrap(),
        ];
        let intersects = intersects(&valid_actions, &excludes_actions);

        assert!(!intersects);
        assert_eq!(valid_actions.len(), 64 - excludes_actions.len());
    }

    #[test]
    fn test_get_valid_horizontal_wall_actions_blocking_path_edge() {
        let mut game_state = GameState::initial();
        game_state.take_action(&"e1v".parse::<Action>().unwrap());
        game_state.take_action(&"e2h".parse::<Action>().unwrap());
        game_state.take_action(&"c2h".parse::<Action>().unwrap());
        game_state.take_action(&"b3v".parse::<Action>().unwrap());

        let valid_actions = game_state
            .get_valid_horizontal_wall_actions()
            .collect::<Vec<_>>();
        let excludes_actions = vec![
            "a2h".parse::<Action>().unwrap(),
            "a3h".parse::<Action>().unwrap(),
            "a4h".parse::<Action>().unwrap(),
            "b2h".parse::<Action>().unwrap(),
            "b3h".parse::<Action>().unwrap(),
            "c2h".parse::<Action>().unwrap(),
            "d2h".parse::<Action>().unwrap(),
            "e2h".parse::<Action>().unwrap(),
            "e1h".parse::<Action>().unwrap(),
            "f2h".parse::<Action>().unwrap(),
        ];
        let intersects = intersects(&valid_actions, &excludes_actions);

        assert!(!intersects);
        assert_eq!(valid_actions.len(), 64 - excludes_actions.len());
    }

    #[test]
    fn test_get_valid_vertical_wall_actions_initial() {
        let game_state = GameState::initial();
        let valid_actions = game_state
            .get_valid_vertical_wall_actions()
            .collect::<Vec<_>>();

        let mut cols = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];
        let rows = [1, 2, 3, 4, 5, 6, 7, 8];
        cols.reverse();

        let mut actions = Vec::new();

        for row in rows.iter() {
            for col in cols.iter() {
                actions.push(format!("{}{}v", col, row).parse().unwrap());
            }
        }

        assert_eq!(valid_actions.len(), actions.len());
        assert_eq!(valid_actions, actions);
    }

    #[test]
    fn test_get_valid_vertical_wall_actions_on_vertical_wall() {
        let mut game_state = GameState::initial();
        game_state.take_action(&"e5v".parse::<Action>().unwrap());

        let valid_actions = game_state
            .get_valid_vertical_wall_actions()
            .collect::<Vec<_>>();
        let excludes_actions = vec![
            "e4v".parse::<Action>().unwrap(),
            "e5v".parse::<Action>().unwrap(),
            "e6v".parse::<Action>().unwrap(),
        ];
        let intersects = intersects(&valid_actions, &excludes_actions);

        assert!(!intersects);
        assert_eq!(valid_actions.len(), 64 - excludes_actions.len());
    }

    #[test]
    fn test_get_valid_vertical_wall_actions_on_horizontal_wall() {
        let mut game_state = GameState::initial();
        game_state.take_action(&"e5h".parse::<Action>().unwrap());

        let valid_actions = game_state
            .get_valid_vertical_wall_actions()
            .collect::<Vec<_>>();
        let excludes_actions = vec!["e5v".parse::<Action>().unwrap()];
        let intersects = intersects(&valid_actions, &excludes_actions);

        assert!(!intersects);
        assert_eq!(valid_actions.len(), 64 - excludes_actions.len());
    }

    #[test]
    fn test_get_valid_wall_actions_on_all_walls_placed() {
        let mut game_state = GameState::initial();
        game_state.take_action(&"a1h".parse::<Action>().unwrap());
        game_state.take_action(&"f9".parse::<Action>().unwrap());
        game_state.take_action(&"c1h".parse::<Action>().unwrap());
        game_state.take_action(&"e9".parse::<Action>().unwrap());
        game_state.take_action(&"e1h".parse::<Action>().unwrap());
        game_state.take_action(&"f9".parse::<Action>().unwrap());
        game_state.take_action(&"g1h".parse::<Action>().unwrap());
        game_state.take_action(&"e9".parse::<Action>().unwrap());
        game_state.take_action(&"a2h".parse::<Action>().unwrap());
        game_state.take_action(&"f9".parse::<Action>().unwrap());
        game_state.take_action(&"c2h".parse::<Action>().unwrap());
        game_state.take_action(&"e9".parse::<Action>().unwrap());
        game_state.take_action(&"e2h".parse::<Action>().unwrap());
        game_state.take_action(&"f9".parse::<Action>().unwrap());
        game_state.take_action(&"g2h".parse::<Action>().unwrap());
        game_state.take_action(&"e9".parse::<Action>().unwrap());
        game_state.take_action(&"a3h".parse::<Action>().unwrap());
        game_state.take_action(&"f9".parse::<Action>().unwrap());

        // 9 walls placed
        let valid_actions = game_state.get_valid_horizontal_wall_actions();
        assert_eq!(valid_actions.count(), 46);

        game_state.take_action(&"c3h".parse::<Action>().unwrap());
        game_state.take_action(&"e9".parse::<Action>().unwrap());

        // 10 walls placed so we shouldn't be able to place anymore, horizontal or vertical
        let valid_horizontal_actions = game_state.get_valid_horizontal_wall_actions();
        assert_eq!(valid_horizontal_actions.count(), 0);

        let valid_vertical_actions = game_state.get_valid_vertical_wall_actions();
        assert_eq!(valid_vertical_actions.count(), 0);
    }

    #[test]
    fn test_is_terminal_p2() {
        let mut game_state = GameState::initial();
        game_state.take_action(&"e2".parse::<Action>().unwrap());
        game_state.take_action(&"e8".parse::<Action>().unwrap());
        game_state.take_action(&"e3".parse::<Action>().unwrap());
        game_state.take_action(&"e7".parse::<Action>().unwrap());
        game_state.take_action(&"e4".parse::<Action>().unwrap());
        game_state.take_action(&"e6".parse::<Action>().unwrap());
        game_state.take_action(&"e5".parse::<Action>().unwrap());
        game_state.take_action(&"e4".parse::<Action>().unwrap());
        game_state.take_action(&"e6".parse::<Action>().unwrap());
        game_state.take_action(&"e3".parse::<Action>().unwrap());
        game_state.take_action(&"e7".parse::<Action>().unwrap());
        game_state.take_action(&"e2".parse::<Action>().unwrap());
        game_state.take_action(&"e8".parse::<Action>().unwrap());

        let is_terminal = game_state.is_terminal();
        assert_eq!(is_terminal, None);

        game_state.take_action(&"e1".parse::<Action>().unwrap());

        let is_terminal = game_state.is_terminal();
        assert_eq!(is_terminal, Some(Value([0.0, 1.0])));
    }

    #[test]
    fn test_is_terminal_p1() {
        let mut game_state = GameState::initial();
        game_state.take_action(&"e2".parse::<Action>().unwrap());
        game_state.take_action(&"e8".parse::<Action>().unwrap());
        game_state.take_action(&"e3".parse::<Action>().unwrap());
        game_state.take_action(&"e7".parse::<Action>().unwrap());
        game_state.take_action(&"e4".parse::<Action>().unwrap());
        game_state.take_action(&"e6".parse::<Action>().unwrap());
        game_state.take_action(&"e5".parse::<Action>().unwrap());
        game_state.take_action(&"e4".parse::<Action>().unwrap());
        game_state.take_action(&"e6".parse::<Action>().unwrap());
        game_state.take_action(&"e3".parse::<Action>().unwrap());
        game_state.take_action(&"e7".parse::<Action>().unwrap());
        game_state.take_action(&"e2".parse::<Action>().unwrap());
        game_state.take_action(&"e8".parse::<Action>().unwrap());

        let is_terminal = game_state.is_terminal();
        assert_eq!(is_terminal, None);

        game_state.take_action(&"e3".parse::<Action>().unwrap());
        game_state.take_action(&"e9".parse::<Action>().unwrap());

        let is_terminal = game_state.is_terminal();
        assert_eq!(is_terminal, Some(Value([1.0, 0.0])));
    }
}
