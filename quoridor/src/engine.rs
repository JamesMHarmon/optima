use std::fmt::{self,Display,Formatter};
use super::constants::{BOARD_WIDTH,BOARD_HEIGHT,NUM_WALLS_PER_PLAYER,MAX_NUMBER_OF_MOVES,ASCII_LETTER_A};
use super::action::Coordinate;
use super::action::{Action};
use super::board::{map_board_to_arr_invertable,BoardType};
use engine::engine::GameEngine;
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

#[derive(Hash, Eq, PartialEq, Clone, Debug)]
pub struct GameState {
    pub num_moves: usize,
    pub p1_turn_to_move: bool,
    pub p1_pawn_board: u128,
    pub p2_pawn_board: u128,
    pub p1_num_walls_placed: usize,
    pub p2_num_walls_placed: usize,
    pub vertical_wall_placement_board: u128,
    pub horizontal_wall_placement_board: u128
}

impl Display for GameState {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let p1_board = map_board_to_arr_invertable(self.p1_pawn_board, BoardType::Pawn, false);
        let p2_board = map_board_to_arr_invertable(self.p2_pawn_board, BoardType::Pawn, false);
        let horizontal_wall_placement = map_board_to_arr_invertable(self.horizontal_wall_placement_board, BoardType::Pawn, false);
        let vertical_wall_placement = map_board_to_arr_invertable(self.vertical_wall_placement_board, BoardType::Pawn, false);
        let horizontal_wall_board = map_board_to_arr_invertable(self.horizontal_wall_placement_board, BoardType::HorizontalWall, false);
        let vertical_wall_board = map_board_to_arr_invertable(self.vertical_wall_placement_board, BoardType::VerticalWall, false);

        writeln!(f, "")?;

        for y in 0..BOARD_HEIGHT {
            for x in 0..BOARD_WIDTH {
                if x == 0 { write!(f, "  +")?; }
                let idx = y * BOARD_WIDTH + x;
                let w = if horizontal_wall_board[idx] != 0.0 { "■■■" } else { "---" };
                let c = if horizontal_wall_placement[idx] != 0.0 { "■" } else if vertical_wall_placement[idx] != 0.0 { "█" } else { "+" };
                write!(f, "{}{}", w, c)?;
            }

            writeln!(f, "")?;

            for x in 0..BOARD_WIDTH {
                let idx = y * BOARD_WIDTH + x;
                if x == 0 { write!(f, "{} |", BOARD_HEIGHT - y)?; }
                let p = if p1_board[idx] != 0.0 { "1" } else if p2_board[idx] != 0.0 { "2" } else { " " };
                let w = if vertical_wall_board[idx] != 0.0 { "█" } else { "|" };
                write!(f, " {} {}", p, w)?;
            }

            writeln!(f, "")?;
        }

        for x in 0..BOARD_WIDTH {
            if x == 0 { write!(f, "  +")?; }
            write!(f, "---+")?;
        }

        writeln!(f, "")?;

        for x in 0..BOARD_WIDTH {
            if x == 0 { write!(f, "   ")?; }
            let col_letter = (ASCII_LETTER_A + x as u8) as char;
            write!(f, " {}  ", col_letter)?;
        }

        writeln!(f, "")?;
        writeln!(f, "")?;
        writeln!(f, "  P1: {}  P2: {}", NUM_WALLS_PER_PLAYER - self.p1_num_walls_placed, NUM_WALLS_PER_PLAYER - self.p2_num_walls_placed)?;

        Ok(())
    }
}

type Value = [f32; 2];

#[derive(Debug)]
struct PathingResult {
    has_path: bool,
    path: u128
}

impl GameState {
    pub fn take_action(&self, action: &Action) -> Self {
        match action {
            Action::MovePawn(coord) => self.move_pawn(coord.as_bit_board()),
            Action::PlaceHorizontalWall(coord) => self.place_horizontal_wall(coord.as_bit_board()),
            Action::PlaceVerticalWall(coord) => self.place_vertical_wall(coord.as_bit_board())
        }
    }

    pub fn get_valid_pawn_move_actions(&self) -> Vec<Action> {
        Self::map_bit_board_to_coordinates(self.get_valid_pawn_moves())
            .into_iter()
            .map(|coord| Action::MovePawn(coord))
            .collect()
    }

    pub fn get_valid_horizontal_wall_actions(&self) -> Vec<Action> {
        Self::map_bit_board_to_coordinates(self.get_valid_horizontal_wall_placement())
            .into_iter()
            .map(|coord| Action::PlaceHorizontalWall(coord))
            .collect()
    }

    pub fn get_valid_vertical_wall_actions(&self) -> Vec<Action> {
        Self::map_bit_board_to_coordinates(self.get_valid_vertical_wall_placement())
            .into_iter()
            .map(|coord| Action::PlaceVerticalWall(coord))
            .collect()
    }

    pub fn is_terminal(&self) -> Option<Value> {
        let pawn_board = if self.p1_turn_to_move { self.p2_pawn_board } else { self.p1_pawn_board };
        let objective_mask = if self.p1_turn_to_move { P2_OBJECTIVE_MASK } else { P1_OBJECTIVE_MASK };

        if pawn_board & objective_mask != 0 {
            Some(if self.p1_turn_to_move { [0.0, 1.0] } else { [1.0, 0.0] })
        } else if self.num_moves >= MAX_NUMBER_OF_MOVES {
            // A game that runs too long will be a loss for both players.
            Some([-1.0, -1.0])
        }
        else {
            None
        }
    }

    fn move_pawn(&self, pawn_board: u128) -> Self {
        let p1_turn_to_move = self.p1_turn_to_move;

        Self {
            num_moves: self.num_moves + 1,
            p1_turn_to_move: !p1_turn_to_move,
            p1_pawn_board: if p1_turn_to_move { pawn_board } else { self.p1_pawn_board },
            p2_pawn_board: if !p1_turn_to_move { pawn_board } else { self.p2_pawn_board },
            ..*self
        }
    }

    fn place_horizontal_wall(&self, horizontal_wall_placement: u128) -> Self {
        let p1_turn_to_move = self.p1_turn_to_move;

        Self {
            num_moves: self.num_moves + 1,
            p1_turn_to_move: !p1_turn_to_move,
            p1_num_walls_placed: self.p1_num_walls_placed + if p1_turn_to_move { 1 } else { 0 },
            p2_num_walls_placed: self.p2_num_walls_placed + if !p1_turn_to_move { 1 } else { 0 },
            horizontal_wall_placement_board: self.horizontal_wall_placement_board | horizontal_wall_placement,
            ..*self
        }
    }

    fn place_vertical_wall(&self, vertical_wall_placement: u128) -> Self {
        let p1_turn_to_move = self.p1_turn_to_move;

        Self {
            num_moves: self.num_moves + 1,
            p1_turn_to_move: !p1_turn_to_move,
            p1_num_walls_placed: self.p1_num_walls_placed + if p1_turn_to_move { 1 } else { 0 },
            p2_num_walls_placed: self.p2_num_walls_placed + if !p1_turn_to_move { 1 } else { 0 },
            vertical_wall_placement_board: self.vertical_wall_placement_board | vertical_wall_placement,
            ..*self
        }
    }

    fn get_valid_pawn_moves(&self) -> u128 {
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

        let straight_jump_move = straight_jump_up_move | straight_jump_right_move | straight_jump_down_move | straight_jump_left_move;

        if straight_jump_move != 0 {
            return valid_moves & !opposing_player_board | straight_jump_move;
        }

        let side_jump_moves =
            (
                shift_up!(overlapping_move) & move_up_mask
                | shift_right!(overlapping_move)& move_right_mask
                | shift_down!(overlapping_move) & move_down_mask
                | shift_left!(overlapping_move) & move_left_mask
            ) 
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
        shift_up!(self.vertical_wall_placement_board) | self.vertical_wall_placement_board
    }

    fn get_horizontal_wall_blocks(&self) -> u128 {
        shift_right!(self.horizontal_wall_placement_board) | self.horizontal_wall_placement_board
    }

    fn get_valid_horizontal_wall_placement(&self) -> u128 {
        if self.active_player_has_wall_to_place() {
            self.get_valid_horizontal_wall_positions()
        } else {
            0
        }
    }

    fn get_valid_vertical_wall_placement(&self) -> u128 {
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
        !(self.horizontal_wall_placement_board | shift_right!(self.horizontal_wall_placement_board) | shift_left!(self.horizontal_wall_placement_board))
        & !self.vertical_wall_placement_board
        & CANDIDATE_WALL_PLACEMENT_MASK
    }

    fn get_candidate_vertical_wall_placement(&self) -> u128 {
        !(self.vertical_wall_placement_board | shift_down!(self.vertical_wall_placement_board) | shift_up!(self.vertical_wall_placement_board))
        & !self.horizontal_wall_placement_board
        & CANDIDATE_WALL_PLACEMENT_MASK
    }

    fn get_invalid_horizontal_wall_candidates(&self) -> u128 {
        let mut invalid_placements: u128 = 0;
        let mut horizontal_connecting_candidates = self.get_horizontal_connecting_candidates();

        while horizontal_connecting_candidates != 0 {
            let with_removed_candidate = horizontal_connecting_candidates & (horizontal_connecting_candidates - 1);
            let removed_candidate = with_removed_candidate ^ horizontal_connecting_candidates;

            let state_with_candidate = Self {
                horizontal_wall_placement_board: self.horizontal_wall_placement_board | removed_candidate,
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
            let with_removed_candidate = vertical_connecting_candidates & (vertical_connecting_candidates - 1);
            let removed_candidate = with_removed_candidate ^ vertical_connecting_candidates;

            let state_with_candidate = Self {
                vertical_wall_placement_board: self.vertical_wall_placement_board | removed_candidate,
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
        let p2_path_start = if p1_path_result.path & self.p2_pawn_board != 0 { p1_path_result.path } else { self.p2_pawn_board };

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
                return PathingResult { has_path: true, path: updated_path };
            }

            // Check if any progress was made, if there is no progress from the last iteration then we are stuck.
            if updated_path == path {
                return PathingResult { has_path: false, path: updated_path };
            }

            path = updated_path;
        }
    }

    fn get_move_up_mask(&self) -> u128 {
        !(shift_up!(self.get_horizontal_wall_blocks())) & VALID_PIECE_POSITION_MASK
    }

    fn get_move_right_mask(&self) -> u128 {
        !shift_right!(self.get_vertical_wall_blocks()) & VALID_PIECE_POSITION_MASK & !LEFT_COLUMN_MASK
    }

    fn get_move_down_mask(&self) -> u128{
        !self.get_horizontal_wall_blocks() & VALID_PIECE_POSITION_MASK
    }

    fn get_move_left_mask(&self) -> u128 {
        !self.get_vertical_wall_blocks() & VALID_PIECE_POSITION_MASK & !RIGHT_COLUMN_MASK
    }

    fn get_horizontal_connecting_candidates(&self) -> u128 {
        let candidate_horizontal_walls = self.get_candidate_horizontal_wall_placement();
        let horizontal_walls = self.horizontal_wall_placement_board;
        let vertical_walls = self.vertical_wall_placement_board;

        // Compare to existing walls. Wall segment candidates check against locations of possible new connections. These are checked in combinations since the walls may already be connected. We don't want to count the same existing connection twice or thrice.
        let middle_touching = candidate_horizontal_walls & (shift_up!(vertical_walls) | shift_down!(vertical_walls));
        let left_edge_touching = candidate_horizontal_walls & (shift_right!(vertical_walls) | shift_down_right!(vertical_walls) | shift_up_right!(vertical_walls) | shift_right!(shift_right!(horizontal_walls)) | HORIZONTAL_WALL_LEFT_EDGE_TOUCHING_BOARD_MASK);
        let right_edge_touching = candidate_horizontal_walls & (shift_left!(vertical_walls) | shift_down_left!(vertical_walls) | shift_up_left!(vertical_walls) | shift_left!(shift_left!(horizontal_walls)) | HORIZONTAL_WALL_RIGHT_EDGE_TOUCHING_BOARD_MASK);

        (left_edge_touching & middle_touching) | (left_edge_touching & right_edge_touching) | (middle_touching & right_edge_touching)
    }

    fn get_vertical_connecting_candidates(&self) -> u128 {
        let candidate_vertical_walls = self.get_candidate_vertical_wall_placement();
        let vertical_walls = self.vertical_wall_placement_board;
        let horizontal_walls = self.horizontal_wall_placement_board;

        // Compare to existing walls. Wall segment candidates check against locations of possible new connections. These are checked in combinations since the walls may already be connected. We don't want to count the same existing connection twice or thrice.
        let middle_touching = candidate_vertical_walls & (shift_right!(horizontal_walls) | shift_left!(horizontal_walls));
        let top_edge_touching = candidate_vertical_walls & (shift_down_left!(horizontal_walls) | shift_down!(horizontal_walls) | shift_down_right!(horizontal_walls) | shift_down!(shift_down!(vertical_walls))| VERTICAL_WALL_TOP_EDGE_TOUCHING_BOARD_MASK);
        let bottom_edge_touching = candidate_vertical_walls & (shift_up_left!(horizontal_walls) | shift_up!(horizontal_walls) | shift_up_right!(horizontal_walls) | shift_up!(shift_up!(vertical_walls)) | VERTICAL_WALL_BOTTOM_EDGE_TOUCHING_BOARD_MASK);

        (top_edge_touching & middle_touching) | (top_edge_touching & bottom_edge_touching) | (middle_touching & bottom_edge_touching)
    }

    fn map_bit_board_to_coordinates(board: u128) -> Vec<Coordinate> {
        let mut board = board;
        let mut coordinates = Vec::new();

        while board != 0 {
            let board_without_first_bit = board & (board - 1);
            let removed_bit = board_without_first_bit ^ board;
            let coordinate = Coordinate::from_bit_board(removed_bit);
            coordinates.push(coordinate);

            board = board_without_first_bit;
        }

        coordinates
    }
}

impl game_state::GameState for GameState {
    fn initial() -> Self {
        GameState {
            num_moves: 0,
            p1_turn_to_move: true,
            p1_pawn_board: P1_STARTING_POS_MASK,
            p2_pawn_board: P2_STARTING_POS_MASK,
            p1_num_walls_placed: 0,
            p2_num_walls_placed: 0,
            vertical_wall_placement_board: 0,
            horizontal_wall_placement_board: 0
        }
    }
}

pub struct Engine {}

impl Engine {
    pub fn new() -> Self { Self {} }
}

impl GameEngine for Engine {
    type Action = Action;
    type State = GameState;
    type Value = Value;

    fn take_action(&self, game_state: &GameState, action: &Action) -> GameState {
        game_state.take_action(action)
    }

    fn is_terminal_state(&self, game_state: &GameState) -> Option<Self::Value> {
        game_state.is_terminal()
    }
}

#[cfg(test)]
mod tests {
    use super::GameState;
    use super::super::action::{Action,Coordinate};
    use engine::game_state::{GameState as GameStateTrait};

    fn intersects(actions: &Vec<Action>, exclusions: &Vec<Action>) -> bool {
        actions.iter().any(|a| exclusions.iter().any(|a2| a == a2))
    }

    #[test]
    fn test_get_valid_pawn_move_actions_p1() {
        let game_state = GameState::initial();
        let valid_actions = game_state.get_valid_pawn_move_actions();

        assert_eq!(valid_actions, vec!(
            Action::MovePawn(Coordinate::new('f', 1)),
            Action::MovePawn(Coordinate::new('d', 1)),
            Action::MovePawn(Coordinate::new('e', 2))
        ));
    }

    #[test]
    fn test_get_valid_pawn_move_actions_p2() {
        let game_state = GameState::initial();
        let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('f',1)));
        let valid_actions = game_state.get_valid_pawn_move_actions();

        assert_eq!(valid_actions, vec!(
            Action::MovePawn(Coordinate::new('e', 8)),
            Action::MovePawn(Coordinate::new('f', 9)),
            Action::MovePawn(Coordinate::new('d', 9))
        ));
    }

    #[test]
    fn test_get_valid_pawn_move_actions_vertical_wall() {
        let game_state = GameState::initial();
        let game_state = game_state.take_action(&Action::PlaceVerticalWall(Coordinate::new('d',1)));
        let game_state = game_state.take_action(&Action::PlaceVerticalWall(Coordinate::new('e',1)));
        let valid_actions = game_state.get_valid_pawn_move_actions();

        assert_eq!(valid_actions, vec!(
            Action::MovePawn(Coordinate::new('e', 2))
        ));
    }

    #[test]
    fn test_get_valid_pawn_move_actions_vertical_wall_top() {
        let game_state = GameState::initial();
        let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('e',2)));
        let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('e',7)));
        let game_state = game_state.take_action(&Action::PlaceVerticalWall(Coordinate::new('d',1)));
        let game_state = game_state.take_action(&Action::PlaceVerticalWall(Coordinate::new('e',1)));
        let valid_actions = game_state.get_valid_pawn_move_actions();

        assert_eq!(valid_actions, vec!(
            Action::MovePawn(Coordinate::new('e', 1)),
            Action::MovePawn(Coordinate::new('e', 3))
        ));
    }

    #[test]
    fn test_get_valid_pawn_move_actions_horizontal_wall() {
        let game_state = GameState::initial();
        let game_state = game_state.take_action(&Action::PlaceHorizontalWall(Coordinate::new('d',8)));
        let game_state = game_state.take_action(&Action::PlaceHorizontalWall(Coordinate::new('e',1)));
        let valid_actions = game_state.get_valid_pawn_move_actions();

        assert_eq!(valid_actions, vec!(
            Action::MovePawn(Coordinate::new('f', 1)),
            Action::MovePawn(Coordinate::new('d', 1))
        ));

        let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('f',1)));
        let valid_actions = game_state.get_valid_pawn_move_actions();

        assert_eq!(valid_actions, vec!(
            Action::MovePawn(Coordinate::new('f', 9)),
            Action::MovePawn(Coordinate::new('d', 9))
        ));

        let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('f',9)));
        let valid_actions = game_state.get_valid_pawn_move_actions();

        assert_eq!(valid_actions, vec!(
            Action::MovePawn(Coordinate::new('g', 1)),
            Action::MovePawn(Coordinate::new('e', 1))
        ));
    }

    #[test]
    fn test_get_valid_pawn_move_actions_blocked() {
        let game_state = GameState::initial();
        let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('e',2)));
        let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('e',8)));
        let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('e',3)));
        let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('e',7)));
        let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('e',4)));
        let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('e',6)));
        let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('e',5)));

        let valid_actions = game_state.get_valid_pawn_move_actions();
        assert_eq!(valid_actions, vec!(
            Action::MovePawn(Coordinate::new('e',4)),
            Action::MovePawn(Coordinate::new('f',6)),
            Action::MovePawn(Coordinate::new('d',6)),
            Action::MovePawn(Coordinate::new('e',7))
        ));

        let game_state = game_state.take_action(&Action::PlaceHorizontalWall(Coordinate::new('e',4)));
        let game_state = game_state.take_action(&Action::PlaceHorizontalWall(Coordinate::new('a',1)));
        let valid_actions = game_state.get_valid_pawn_move_actions();

        assert_eq!(valid_actions, vec!(
            Action::MovePawn(Coordinate::new('f',5)),
            Action::MovePawn(Coordinate::new('d',5)),
            Action::MovePawn(Coordinate::new('f',6)),
            Action::MovePawn(Coordinate::new('d',6)),
            Action::MovePawn(Coordinate::new('e',7))
        ));

        let game_state = game_state.take_action(&Action::PlaceHorizontalWall(Coordinate::new('e',6)));
        let valid_actions = game_state.get_valid_pawn_move_actions();

        assert_eq!(valid_actions, vec!(
            Action::MovePawn(Coordinate::new('f',5)),
            Action::MovePawn(Coordinate::new('d',5)),
            Action::MovePawn(Coordinate::new('f',6)),
            Action::MovePawn(Coordinate::new('d',6))
        ));
    }

    #[test]
    fn test_get_valid_horizontal_wall_actions_initial() {
        let game_state = GameState::initial();
        let valid_actions = game_state.get_valid_horizontal_wall_actions();

        let mut cols = ['a','b','c','d','e','f','g','h'];
        let rows = [1,2,3,4,5,6,7,8];
        cols.reverse();

        let mut actions = Vec::new();

        for row in rows.into_iter() {
            for col in cols.into_iter() {
                actions.push(Action::PlaceHorizontalWall(Coordinate::new(*col, *row)));
            }
        }

        assert_eq!(valid_actions.len(), actions.len());
        assert_eq!(valid_actions, actions);
    }

    #[test]
    fn test_get_valid_horizontal_wall_actions_on_horizontal_wall() {
        let game_state = GameState::initial();
        let game_state = game_state.take_action(&Action::PlaceHorizontalWall(Coordinate::new('d',1)));

        let valid_actions = game_state.get_valid_horizontal_wall_actions();
        let excludes_actions = vec!(
            Action::PlaceHorizontalWall(Coordinate::new('c', 1)),
            Action::PlaceHorizontalWall(Coordinate::new('d', 1)),
            Action::PlaceHorizontalWall(Coordinate::new('e', 1))
        );
        let intersects = intersects(&valid_actions, &excludes_actions);

        assert_eq!(intersects, false);
        assert_eq!(valid_actions.len(), 64 - excludes_actions.len());
    }

    #[test]
    fn test_get_valid_horizontal_wall_actions_on_vertical_wall() {
        let game_state = GameState::initial();
        let game_state = game_state.take_action(&Action::PlaceVerticalWall(Coordinate::new('e',5)));

        let valid_actions = game_state.get_valid_horizontal_wall_actions();
        let excludes_actions = vec!(
            Action::PlaceHorizontalWall(Coordinate::new('e', 5))
        );
        let intersects = intersects(&valid_actions, &excludes_actions);

        assert_eq!(intersects, false);
        assert_eq!(valid_actions.len(), 64 - excludes_actions.len());
    }

    #[test]
    fn test_get_valid_horizontal_wall_actions_blocking_path() {
        let game_state = GameState::initial();
        let game_state = game_state.take_action(&Action::PlaceVerticalWall(Coordinate::new('c',1)));
        let game_state = game_state.take_action(&Action::PlaceVerticalWall(Coordinate::new('e',1)));

        let valid_actions = game_state.get_valid_horizontal_wall_actions();
        let excludes_actions = vec!(
            Action::PlaceHorizontalWall(Coordinate::new('c', 1)),
            Action::PlaceHorizontalWall(Coordinate::new('e', 1)),
            Action::PlaceHorizontalWall(Coordinate::new('d', 1)),
            Action::PlaceHorizontalWall(Coordinate::new('d', 2))
        );
        let intersects = intersects(&valid_actions, &excludes_actions);

        assert_eq!(intersects, false);
        assert_eq!(valid_actions.len(), 64 - excludes_actions.len());
    }

    #[test]
    fn test_get_valid_horizontal_wall_actions_blocking_path_other_player() {
        let game_state = GameState::initial();
        let game_state = game_state.take_action(&Action::PlaceVerticalWall(Coordinate::new('c',1)));
        let game_state = game_state.take_action(&Action::PlaceVerticalWall(Coordinate::new('e',1)));
        let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('e',2)));

        let valid_actions = game_state.get_valid_horizontal_wall_actions();
        let excludes_actions = vec!(
            Action::PlaceHorizontalWall(Coordinate::new('c', 1)),
            Action::PlaceHorizontalWall(Coordinate::new('e', 1)),
            Action::PlaceHorizontalWall(Coordinate::new('d', 2))
        );
        let intersects = intersects(&valid_actions, &excludes_actions);

        assert_eq!(intersects, false);
        assert_eq!(valid_actions.len(), 64 - excludes_actions.len());
    }

    #[test]
    fn test_get_valid_horizontal_wall_actions_blocking_path_vert_horz() {
        let game_state = GameState::initial();
        let game_state = game_state.take_action(&Action::PlaceVerticalWall(Coordinate::new('c',1)));
        let game_state = game_state.take_action(&Action::PlaceVerticalWall(Coordinate::new('e',1)));
        let game_state = game_state.take_action(&Action::PlaceHorizontalWall(Coordinate::new('c',2)));

        let valid_actions = game_state.get_valid_horizontal_wall_actions();
        let excludes_actions = vec!(
            Action::PlaceHorizontalWall(Coordinate::new('c', 1)),
            Action::PlaceHorizontalWall(Coordinate::new('e', 1)),
            Action::PlaceHorizontalWall(Coordinate::new('d', 2)),
            Action::PlaceHorizontalWall(Coordinate::new('b', 2)),
            Action::PlaceHorizontalWall(Coordinate::new('c', 2)),
            Action::PlaceHorizontalWall(Coordinate::new('d', 2)),
            Action::PlaceHorizontalWall(Coordinate::new('e', 2)),
        );
        let intersects = intersects(&valid_actions, &excludes_actions);

        assert_eq!(intersects, false);
        assert_eq!(valid_actions.len(), 64 - excludes_actions.len());
    }

    #[test]
    fn test_get_valid_horizontal_wall_actions_blocking_path_edge() {
        let game_state = GameState::initial();
        let game_state = game_state.take_action(&Action::PlaceVerticalWall(Coordinate::new('e',1)));
        let game_state = game_state.take_action(&Action::PlaceHorizontalWall(Coordinate::new('e',2)));
        let game_state = game_state.take_action(&Action::PlaceHorizontalWall(Coordinate::new('c',2)));
        let game_state = game_state.take_action(&Action::PlaceVerticalWall(Coordinate::new('b',3)));

        let valid_actions = game_state.get_valid_horizontal_wall_actions();
        let excludes_actions = vec!(
            Action::PlaceHorizontalWall(Coordinate::new('a', 2)),
            Action::PlaceHorizontalWall(Coordinate::new('a', 3)),
            Action::PlaceHorizontalWall(Coordinate::new('a', 4)),
            Action::PlaceHorizontalWall(Coordinate::new('b', 2)),
            Action::PlaceHorizontalWall(Coordinate::new('b', 3)),
            Action::PlaceHorizontalWall(Coordinate::new('c', 2)),
            Action::PlaceHorizontalWall(Coordinate::new('d', 2)),
            Action::PlaceHorizontalWall(Coordinate::new('e', 2)),
            Action::PlaceHorizontalWall(Coordinate::new('e', 1)),
            Action::PlaceHorizontalWall(Coordinate::new('f', 2)),
        );
        let intersects = intersects(&valid_actions, &excludes_actions);

        assert_eq!(intersects, false);
        assert_eq!(valid_actions.len(), 64 - excludes_actions.len());
    }

    #[test]
    fn test_get_valid_vertical_wall_actions_initial() {
        let game_state = GameState::initial();
        let valid_actions = game_state.get_valid_vertical_wall_actions();

        let mut cols = ['a','b','c','d','e','f','g','h'];
        let rows = [1,2,3,4,5,6,7,8];
        cols.reverse();

        let mut actions = Vec::new();

        for row in rows.into_iter() {
            for col in cols.into_iter() {
                actions.push(Action::PlaceVerticalWall(Coordinate::new(*col, *row)));
            }
        }

        assert_eq!(valid_actions.len(), actions.len());
        assert_eq!(valid_actions, actions);
    }

    #[test]
    fn test_get_valid_vertical_wall_actions_on_vertical_wall() {
        let game_state = GameState::initial();
        let game_state = game_state.take_action(&Action::PlaceVerticalWall(Coordinate::new('e',5)));

        let valid_actions = game_state.get_valid_vertical_wall_actions();
        let excludes_actions = vec!(
            Action::PlaceVerticalWall(Coordinate::new('e', 4)),
            Action::PlaceVerticalWall(Coordinate::new('e', 5)),
            Action::PlaceVerticalWall(Coordinate::new('e', 6))
        );
        let intersects = intersects(&valid_actions, &excludes_actions);

        assert_eq!(intersects, false);
        assert_eq!(valid_actions.len(), 64 - excludes_actions.len());
    }

    #[test]
    fn test_get_valid_vertical_wall_actions_on_horizontal_wall() {
        let game_state = GameState::initial();
        let game_state = game_state.take_action(&Action::PlaceHorizontalWall(Coordinate::new('e',5)));

        let valid_actions = game_state.get_valid_vertical_wall_actions();
        let excludes_actions = vec!(
            Action::PlaceVerticalWall(Coordinate::new('e', 5))
        );
        let intersects = intersects(&valid_actions, &excludes_actions);

        assert_eq!(intersects, false);
        assert_eq!(valid_actions.len(), 64 - excludes_actions.len());
    }

    #[test]
    fn test_get_valid_wall_actions_on_all_walls_placed() {
        let game_state = GameState::initial();
        let game_state = game_state.take_action(&Action::PlaceHorizontalWall(Coordinate::new('a',1)));
        let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('f',9)));
        let game_state = game_state.take_action(&Action::PlaceHorizontalWall(Coordinate::new('c',1)));
        let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('e',9)));
        let game_state = game_state.take_action(&Action::PlaceHorizontalWall(Coordinate::new('e',1)));
        let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('f',9)));
        let game_state = game_state.take_action(&Action::PlaceHorizontalWall(Coordinate::new('g',1)));
        let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('e',9)));
        let game_state = game_state.take_action(&Action::PlaceHorizontalWall(Coordinate::new('a',2)));
        let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('f',9)));
        let game_state = game_state.take_action(&Action::PlaceHorizontalWall(Coordinate::new('c',2)));
        let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('e',9)));
        let game_state = game_state.take_action(&Action::PlaceHorizontalWall(Coordinate::new('e',2)));
        let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('f',9)));
        let game_state = game_state.take_action(&Action::PlaceHorizontalWall(Coordinate::new('g',2)));
        let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('e',9)));
        let game_state = game_state.take_action(&Action::PlaceHorizontalWall(Coordinate::new('a',3)));
        let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('f',9)));

        // 9 walls placed
        let valid_actions = game_state.get_valid_horizontal_wall_actions();
        assert_eq!(valid_actions.len(), 46);

        let game_state = game_state.take_action(&Action::PlaceHorizontalWall(Coordinate::new('c',3)));
        let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('e',9)));

        // 10 walls placed so we shouldn't be able to place anymore, horizontal or vertical
        let valid_horizontal_actions = game_state.get_valid_horizontal_wall_actions();
        assert_eq!(valid_horizontal_actions.len(), 0);

        let valid_vertical_actions = game_state.get_valid_vertical_wall_actions();
        assert_eq!(valid_vertical_actions.len(), 0);
    }

    #[test]
    fn test_is_terminal_p2() {
        let game_state = GameState::initial();
        let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('e',2)));
        let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('e',8)));
        let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('e',3)));
        let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('e',7)));
        let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('e',4)));
        let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('e',6)));
        let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('e',5)));
        let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('e',4)));
        let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('e',6)));
        let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('e',3)));
        let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('e',7)));
        let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('e',2)));
        let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('e',8)));

        let is_terminal = game_state.is_terminal();
        assert_eq!(is_terminal, None);

        let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('e',1)));

        let is_terminal = game_state.is_terminal();
        assert_eq!(is_terminal, Some([0.0, 1.0]));
    }

    #[test]
    fn test_is_terminal_p1() {
        let game_state = GameState::initial();
        let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('e',2)));
        let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('e',8)));
        let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('e',3)));
        let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('e',7)));
        let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('e',4)));
        let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('e',6)));
        let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('e',5)));
        let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('e',4)));
        let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('e',6)));
        let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('e',3)));
        let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('e',7)));
        let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('e',2)));
        let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('e',8)));

        let is_terminal = game_state.is_terminal();
        assert_eq!(is_terminal, None);

        let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('e',3)));
        let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('e',9)));

        let is_terminal = game_state.is_terminal();
        assert_eq!(is_terminal, Some([1.0, 0.0]));
    }
}
