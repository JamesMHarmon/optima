use crate::{ActionType, BOARD_SIZE};

use super::constants::{MAX_NUMBER_OF_MOVES, NUM_WALLS_PER_PLAYER};
use super::{Action, Coordinate, Value, Zobrist};
use common::TranspositionHash;
use engine::game_state;

const LEFT_COLUMN_MASK: u128 =                                  0b__100000000__100000000__100000000__100000000__100000000__100000000__100000000__100000000__100000000;
const RIGHT_COLUMN_MASK: u128 =                                 0b__000000001__000000001__000000001__000000001__000000001__000000001__000000001__000000001__000000001;

const VALID_PIECE_POSITION_MASK: u128 =                         0b__111111111__111111111__111111111__111111111__111111111__111111111__111111111__111111111__111111111;
const CANDIDATE_WALL_PLACEMENT_MASK: u128 =                     0b__111111110__111111110__111111110__111111110__111111110__111111110__111111110__111111110__000000000;
const P1_OBJECTIVE_MASK: u128 =                                 0b__111111111__000000000__000000000__000000000__000000000__000000000__000000000__000000000__000000000;
const P2_OBJECTIVE_MASK: u128 =                                 0b__000000000__000000000__000000000__000000000__000000000__000000000__000000000__000000000__111111111;
const P1_STARTING_POS_MASK: u128 =                              0b__000000000__000000000__000000000__000000000__000000000__000000000__000000000__000000000__000010000;
const P2_STARTING_POS_MASK: u128 =                              0b__000010000__000000000__000000000__000000000__000000000__000000000__000000000__000000000__000000000;

const HORIZONTAL_WALL_LEFT_EDGE_TOUCHING_BOARD_MASK: u128 =     0b__100000000__100000000__100000000__100000000__100000000__100000000__100000000__100000000__000000000;
const HORIZONTAL_WALL_RIGHT_EDGE_TOUCHING_BOARD_MASK: u128 =    0b__000000010__000000010__000000010__000000010__000000010__000000010__000000010__000000010__000000000;
const VERTICAL_WALL_TOP_EDGE_TOUCHING_BOARD_MASK: u128 =        0b__111111110__000000000__000000000__000000000__000000000__000000000__000000000__000000000__000000000;
const VERTICAL_WALL_BOTTOM_EDGE_TOUCHING_BOARD_MASK: u128 =     0b__000000000__000000000__000000000__000000000__000000000__000000000__000000000__111111110__000000000;

#[derive(Clone, Debug)]
pub struct GameState {
    move_number: usize,
    victory_margin: u8,
    is_final: bool,
    p1_turn_to_move: bool,
    p1_pawn_board: u128,
    p2_pawn_board: u128,
    p1_num_walls: u8,
    p2_num_walls: u8,
    vertical_wall_board: u128,
    horizontal_wall_board: u128,
    zobrist: Zobrist,
}

#[derive(Debug)]
struct PathingResult {
    has_path: bool,
    path: u128,
    distance: u8,
}

impl GameState {
    pub fn new(
        horizontal_walls: impl IntoIterator<Item = Coordinate>,
        vertical_walls: impl IntoIterator<Item = Coordinate>,
        player_positions: impl IntoIterator<Item = Coordinate>,
        walls_remaining: impl IntoIterator<Item = usize>,
        p1_turn_to_move: bool,
    ) -> Self {
        let mut player_positions = player_positions.into_iter();
        let p1_pawn_board = player_positions
            .next()
            .expect("Expected p1 position")
            .as_bit_board();
        let p2_pawn_board = player_positions
            .next()
            .expect("Expected p2 position")
            .as_bit_board();
        let mut walls_remaining = walls_remaining.into_iter();
        let p1_num_walls = walls_remaining.next().expect("Expected p1 walls remaining") as u8;
        let p2_num_walls = walls_remaining.next().expect("Expected p2 walls remaining") as u8;
        let vertical_wall_board = vertical_walls
            .into_iter()
            .fold(0u128, |bit_board, coord| bit_board | coord.as_bit_board());
        let horizontal_wall_board = horizontal_walls
            .into_iter()
            .fold(0u128, |bit_board, coord| bit_board | coord.as_bit_board());

        let mut game_state = Self {
            move_number: 0,
            victory_margin: 0,
            is_final: false,
            p1_turn_to_move,
            p1_pawn_board,
            p2_pawn_board,
            p1_num_walls,
            p2_num_walls,
            vertical_wall_board,
            horizontal_wall_board,
            zobrist: Zobrist::initial(),
        };

        game_state.zobrist = Zobrist::from(&game_state);

        game_state
    }

    pub fn take_action(&mut self, action: &Action) {
        match action.action_type() {
            ActionType::PawnMove => self.move_pawn(action.coord()),
            ActionType::VerticalWall => self.place_wall(action.coord(), true),
            ActionType::HorizontalWall => self.place_wall(action.coord(), false),
            ActionType::Pass => (),
        }

        let is_final = self.try_finalize(action);
        let is_goaling_action = self.is_goaling_action(action);

        if !is_final && !is_goaling_action {
            self.increment_turn();
        }
    }

    pub fn valid_actions(&self) -> impl Iterator<Item = Action> {
        let pawn_moves = self.valid_pawn_move_actions();
        let vertical_walls = self.valid_vertical_wall_actions();
        let horizontal_walls = self.valid_horizontal_wall_actions();

        let pass_action = if self.can_pass() {
            Some(Action::pass()).into_iter()
        } else {
            None.into_iter()
        };

        pawn_moves
            .chain(vertical_walls)
            .chain(horizontal_walls)
            .chain(pass_action)
    }

    fn valid_pawn_move_actions(&self) -> impl Iterator<Item = Action> {
        Self::bit_board_actions(self.valid_pawn_moves(), ActionType::PawnMove)
    }

    fn valid_horizontal_wall_actions(&self) -> impl Iterator<Item = Action> {
        Self::bit_board_actions(
            self.valid_horizontal_wall_placement(),
            ActionType::HorizontalWall,
        )
    }

    fn valid_vertical_wall_actions(&self) -> impl Iterator<Item = Action> {
        Self::bit_board_actions(
            self.valid_vertical_wall_placement(),
            ActionType::VerticalWall,
        )
    }

    pub fn can_pass(&self) -> bool {
        if !self.is_scoring_phase() {
            return false;
        }

        self.p1_turn_to_move && self.is_player_one_at_goal()
            || !self.p1_turn_to_move && self.is_player_two_at_goal()
    }

    pub fn is_terminal(&self) -> Option<Value> {
        if self.is_final {
            Some(if self.p1_turn_to_move {
                Value([0.0, 1.0])
            } else {
                Value([1.0, 0.0])
            })
        } else if self.move_number > MAX_NUMBER_OF_MOVES {
            Some(Value([0.5, 0.5]))
        } else {
            None
        }
    }

    pub fn vertical_symmetry(&self) -> Self {
        let vertical_symmetry_bit_board = |bit_board: u128, shift: bool| {
            Coordinate::from_bit_board_bits(bit_board).fold(0u128, |mut bit_board, coord| {
                bit_board |= coord.vertical_symmetry(shift).as_bit_board();
                bit_board
            })
        };

        let mut symmetrical_state = Self {
            p1_pawn_board: vertical_symmetry_bit_board(self.p1_pawn_board, false),
            p2_pawn_board: vertical_symmetry_bit_board(self.p2_pawn_board, false),
            vertical_wall_board: vertical_symmetry_bit_board(self.vertical_wall_board, true),
            horizontal_wall_board: vertical_symmetry_bit_board(self.horizontal_wall_board, true),
            move_number: self.move_number,
            is_final: self.is_final,
            victory_margin: self.victory_margin,
            p1_num_walls: self.p1_num_walls,
            p2_num_walls: self.p2_num_walls,
            p1_turn_to_move: self.p1_turn_to_move,
            zobrist: Zobrist::initial(),
        };

        symmetrical_state.zobrist = Zobrist::from(&symmetrical_state);

        symmetrical_state
    }

    pub fn transposition_hash(&self) -> u64 {
        self.zobrist.board_state_hash()
    }

    pub fn move_number(&self) -> usize {
        self.move_number
    }

    pub fn p1_turn_to_move(&self) -> bool {
        self.p1_turn_to_move
    }

    pub fn curr_player(&self) -> PlayerInfo {
        let player_num = if self.p1_turn_to_move() { 1 } else { 2 };

        self.player_info(player_num)
    }

    pub fn opp_player(&self) -> PlayerInfo {
        let player_num = if self.p1_turn_to_move() { 2 } else { 1 };

        self.player_info(player_num)
    }

    pub fn player_to_move(&self) -> usize {
        if self.p1_turn_to_move() {
            1
        } else {
            2
        }
    }

    pub fn player_info(&self, player: usize) -> PlayerInfo {
        let (pawn, num_walls, turn) = if player == 1 {
            (
                self.p1_pawn_board.into(),
                self.p1_num_walls as usize,
                self.p1_turn_to_move(),
            )
        } else {
            (
                self.p2_pawn_board.into(),
                self.p2_num_walls as usize,
                !self.p1_turn_to_move(),
            )
        };

        PlayerInfo {
            player_num: player,
            pawn,
            num_walls,
            turn,
        }
    }

    pub fn vertical_walls(&self) -> impl Iterator<Item = Coordinate> {
        Coordinate::from_bit_board_bits(self.vertical_wall_board)
    }

    pub fn horizontal_walls(&self) -> impl Iterator<Item = Coordinate> {
        Coordinate::from_bit_board_bits(self.horizontal_wall_board)
    }

    pub fn victory_margin(&self) -> u8 {
        self.victory_margin
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

        if self.p1_turn_to_move {
            self.p1_num_walls -= 1;
        } else {
            self.p2_num_walls -= 1;
        }

        if is_vertical {
            self.vertical_wall_board |= wall_placement;
        } else {
            self.horizontal_wall_board |= wall_placement;
        }
    }

    /// Finalize the game state and calculate the victory margin.
    /// The game is finalized when either both players have reached their respective goals.
    /// Or when one player is at their goal and that player has no walls remaining.
    /// Or when the winning player chooses to pass.
    fn try_finalize(&mut self, action: &Action) -> bool {
        let player_one_at_goal = self.is_player_one_at_goal();
        let player_two_at_goal = self.is_player_two_at_goal();

        // If both players have reached their goal, the game is over.
        // This means that the victory margin is 0.
        if player_one_at_goal && player_two_at_goal {
            assert!(action.is_move(), "Player must have made a move action.");

            // Do not flip the player to move since the player is the losing player.
            // Margin of victory is 0 if the losing player was able to reach the goal.
            self.victory_margin = 0;
            self.set_is_final();

            return true;
        }

        let p1_at_goal_with_no_walls = self.is_player_one_at_goal() && self.p1_num_walls == 0;
        let p2_at_goal_with_no_walls = self.is_player_two_at_goal() && self.p2_num_walls == 0;
        let player_at_goal_with_no_walls = p1_at_goal_with_no_walls || p2_at_goal_with_no_walls;

        assert!(
            !(p1_at_goal_with_no_walls && p2_at_goal_with_no_walls),
            "Both players cannot be at their goal with no walls remaining."
        );

        // If the winning player is passing, the game is over.
        // Or if the winning player is at their goal and has no walls remaining, the game is over.
        if action.is_pass() || player_at_goal_with_no_walls {
            // Change the player to move since the victor is always the last player to have moved.
            self.toggle_player_turn();
            self.victory_margin = self.curr_player_distance_to_goal();
            self.set_is_final();

            assert!(
                self.victory_margin > 0,
                "Victory margin must be greater than 0."
            );

            assert!(
                self.victory_margin < (BOARD_SIZE as u8),
                "Victory margin must be less than or equal to the board size."
            );

            return true;
        }

        false
    }

    fn set_is_final(&mut self) {
        self.is_final = true;
        self.zobrist = self.zobrist.set_is_final();
    }

    fn increment_turn(&mut self) {
        let curr_player = if self.p1_turn_to_move { 1 } else { 2 };
        let is_scoring_phase = self.is_scoring_phase();

        if curr_player == 2 && !is_scoring_phase {
            self.move_number += 1;
        }

        self.toggle_player_turn();
    }

    fn toggle_player_turn(&mut self) {
        self.p1_turn_to_move = !self.p1_turn_to_move;
        self.zobrist = self.zobrist.toggle_player_turn();
    }

    fn curr_player_distance_to_goal(&self) -> u8 {
        let current_pos_mask = if self.p1_turn_to_move {
            self.p1_pawn_board
        } else {
            self.p2_pawn_board
        };
        let objective_mask = if self.p1_turn_to_move {
            P1_OBJECTIVE_MASK
        } else {
            P2_OBJECTIVE_MASK
        };

        let path = self.find_path(current_pos_mask, objective_mask);

        assert!(path.has_path, "Player has no path to their goal.");

        assert!(
            path.distance < BOARD_SIZE as u8,
            "Distance to goal is greater than the board size."
        );

        path.distance
    }

    fn is_scoring_phase(&self) -> bool {
        self.is_player_one_at_goal() || self.is_player_two_at_goal()
    }

    fn is_player_one_at_goal(&self) -> bool {
        self.p1_pawn_board & P1_OBJECTIVE_MASK != 0
    }

    fn is_player_two_at_goal(&self) -> bool {
        self.p2_pawn_board & P2_OBJECTIVE_MASK != 0
    }

    fn is_goaling_action(&self, action: &Action) -> bool {
        if action.is_move() {
            if self.p1_turn_to_move {
                self.is_player_one_at_goal()
            } else {
                self.is_player_two_at_goal()
            }
        } else {
            false
        }
    }

    fn valid_pawn_moves(&self) -> u128 {
        if self.p1_turn_to_move() && self.is_player_one_at_goal() {
            return 0;
        }

        if !self.p1_turn_to_move() && self.is_player_two_at_goal() {
            return 0;
        }

        let active_player_board = self.active_player_board();
        let opposing_player_board = self.opposing_player_board();

        let move_up_mask = self.move_up_mask();
        let move_right_mask = self.move_right_mask();
        let move_down_mask = self.move_down_mask();
        let move_left_mask = self.move_left_mask();

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

    fn active_player_board(&self) -> u128 {
        if self.p1_turn_to_move {
            self.p1_pawn_board
        } else {
            self.p2_pawn_board
        }
    }

    fn opposing_player_board(&self) -> u128 {
        if self.p1_turn_to_move {
            self.p2_pawn_board
        } else {
            self.p1_pawn_board
        }
    }

    fn vertical_wall_blocks(&self) -> u128 {
        shift_down!(self.vertical_wall_board) | self.vertical_wall_board
    }

    fn horizontal_wall_blocks(&self) -> u128 {
        shift_right!(self.horizontal_wall_board) | self.horizontal_wall_board
    }

    fn valid_horizontal_wall_placement(&self) -> u128 {
        if self.active_player_has_wall_to_place() {
            self.valid_horizontal_wall_positions()
        } else {
            0
        }
    }

    fn valid_vertical_wall_placement(&self) -> u128 {
        if self.active_player_has_wall_to_place() {
            self.valid_vertical_wall_positions()
        } else {
            0
        }
    }

    fn active_player_has_wall_to_place(&self) -> bool {
        if self.p1_turn_to_move {
            self.p1_num_walls > 0
        } else {
            self.p2_num_walls > 0
        }
    }

    fn valid_horizontal_wall_positions(&self) -> u128 {
        let candidate_horizontal_wall_placement = self.candidate_horizontal_wall_placement();
        let valid_horizontal_candidates = !self.invalid_wall_candidates(false);
        candidate_horizontal_wall_placement & valid_horizontal_candidates
    }

    fn valid_vertical_wall_positions(&self) -> u128 {
        let candidate_vertical_wall_placement = self.candidate_vertical_wall_placement();
        let valid_vertical_candidates = !self.invalid_wall_candidates(true);
        candidate_vertical_wall_placement & valid_vertical_candidates
    }

    fn candidate_horizontal_wall_placement(&self) -> u128 {
        !(self.horizontal_wall_board
            | shift_left!(self.horizontal_wall_board)
            | shift_right!(self.horizontal_wall_board))
            & !self.vertical_wall_board
            & CANDIDATE_WALL_PLACEMENT_MASK
    }

    fn candidate_vertical_wall_placement(&self) -> u128 {
        !(self.vertical_wall_board
            | shift_up!(self.vertical_wall_board)
            | shift_down!(self.vertical_wall_board))
            & !self.horizontal_wall_board
            & CANDIDATE_WALL_PLACEMENT_MASK
    }

    fn invalid_wall_candidates(&self, is_vertical: bool) -> u128 {
        let mut invalid_placements: u128 = 0;
        let mut connecting_candidates = if is_vertical {
            self.vertical_connecting_candidates()
        } else {
            self.horizontal_connecting_candidates()
        };

        while connecting_candidates != 0 {
            let excluding_curr_candidate = connecting_candidates & (connecting_candidates - 1);
            let removed_candidate = excluding_curr_candidate ^ connecting_candidates;

            // Clone the current game state so that it can be modified for candidate testing.
            let mut state_with_candidate = self.clone();

            if is_vertical {
                state_with_candidate.vertical_wall_board |= removed_candidate;
            } else {
                state_with_candidate.horizontal_wall_board |= removed_candidate;
            }

            if !state_with_candidate.players_have_path() {
                invalid_placements |= removed_candidate;
            }

            connecting_candidates = excluding_curr_candidate;
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
        let up_mask = self.move_up_mask();
        let right_mask = self.move_right_mask();
        let down_mask = self.move_down_mask();
        let left_mask = self.move_left_mask();

        let mut path = start;
        let mut distance = 0;

        loop {
            // Check if the objective is reachable
            if end & path != 0 {
                return PathingResult {
                    has_path: true,
                    path,
                    distance,
                };
            }

            // MOVE UP & DOWN & LEFT & RIGHT
            let up_path = shift_up!(path) & up_mask;
            let right_path = shift_right!(path) & right_mask;
            let down_path = shift_down!(path) & down_mask;
            let left_path = shift_left!(path) & left_mask;
            let updated_path = up_path | right_path | down_path | left_path | path;

            distance += 1;

            // Check if any progress was made, if there is no progress from the last iteration then we are stuck.
            if updated_path == path {
                return PathingResult {
                    has_path: false,
                    path: updated_path,
                    distance: u8::MAX,
                };
            }

            path = updated_path;
        }
    }

    fn move_up_mask(&self) -> u128 {
        !self.horizontal_wall_blocks() & VALID_PIECE_POSITION_MASK
    }

    fn move_right_mask(&self) -> u128 {
        !shift_right!(self.vertical_wall_blocks()) & VALID_PIECE_POSITION_MASK & !LEFT_COLUMN_MASK
    }

    fn move_down_mask(&self) -> u128 {
        !shift_down!(self.horizontal_wall_blocks()) & VALID_PIECE_POSITION_MASK
    }

    fn move_left_mask(&self) -> u128 {
        !self.vertical_wall_blocks() & VALID_PIECE_POSITION_MASK & !RIGHT_COLUMN_MASK
    }

    fn horizontal_connecting_candidates(&self) -> u128 {
        let candidate_horizontal_walls = self.candidate_horizontal_wall_placement();
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

    fn vertical_connecting_candidates(&self) -> u128 {
        let candidate_vertical_walls = self.candidate_vertical_wall_placement();
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

    fn bit_board_actions(bit_board: u128, action_type: ActionType) -> impl Iterator<Item = Action> {
        Coordinate::from_bit_board_bits(bit_board).map(move |coord| Action::new(action_type, coord))
    }
}

pub struct PlayerInfo {
    player_num: usize,
    num_walls: usize,
    pawn: Coordinate,
    turn: bool,
}

impl PlayerInfo {
    pub fn num_walls(&self) -> usize {
        self.num_walls
    }

    pub fn pawn(&self) -> Coordinate {
        self.pawn
    }

    pub fn turn(&self) -> bool {
        self.turn
    }

    pub fn player_num(&self) -> usize {
        self.player_num
    }
}

impl game_state::GameState for GameState {
    fn initial() -> Self {
        GameState {
            move_number: 1,
            p1_turn_to_move: true,
            p1_pawn_board: P1_STARTING_POS_MASK,
            p2_pawn_board: P2_STARTING_POS_MASK,
            p1_num_walls: NUM_WALLS_PER_PLAYER,
            p2_num_walls: NUM_WALLS_PER_PLAYER,
            vertical_wall_board: 0,
            horizontal_wall_board: 0,
            victory_margin: 0,
            is_final: false,
            zobrist: Zobrist::initial(),
        }
    }
}

impl TranspositionHash for GameState {
    fn transposition_hash(&self) -> u64 {
        self.transposition_hash()
    }
}
