fn main() {
    let game_state = GameState {
        p1_turn_to_move: true,
        p1_num_walls_placed: 0,
        p2_num_walls_placed: 0,
        p1_pawn_board:                     0b__000000000__000010000__000000000__000000000__000000000__000000000__000000000__000000000__000000000__,
        p2_pawn_board:                     0b__000000000__001000000__000000000__000000000__000000000__000000000__000000000__000010000__000000000__,
        vertical_wall_placement_board:     0b__000000000__000000000__000000000__000000000__000000000__000010010__000000000__001000001__000000000__,
        horizontal_wall_placement_board:   0b__000000000__000000000__000000000__000000000__000010100__000000000__000100001__000000000__000101010__,
    };

    let game_state = GameState::new();
    let game_state = game_state.move_pawn(game_state.p1_pawn_board << 9);
    let game_state = game_state.place_horizontal_wall(0b000010000_000000000);
    let game_state = game_state.move_pawn(game_state.p1_pawn_board << 1);


    println!("{}", "Vertical Walls:");
    print_board(game_state.get_vertical_wall_blocks());

    println!("Horizontal Walls:");
    print_board(game_state.get_horizontal_wall_blocks());

    println!("Pawn Positions:");
    print_board(game_state.p1_pawn_board | game_state.p2_pawn_board);

    println!("get_candidate_horizontal_wall_placement:");
    print_board(game_state.get_candidate_horizontal_wall_placement());

    println!("has_path");
    println!("{}", game_state.players_have_path());

    println!("get_horizontal_connecting_candidates");
    print_board(game_state.get_horizontal_connecting_candidates());

    println!("get_invalid_horizontal_wall_candidates");
    print_board(game_state.get_invalid_horizontal_wall_candidates());

    println!("get_valid_horizontal_wall_positions");
    print_board(game_state.get_valid_horizontal_wall_positions());

    println!("Valid Moves");
    print_board(game_state.get_valid_pawn_moves());
}

fn print_board(board: u128) {
    let width = 9;
    let board = format!("{:081b}", board);

    let board = str::replace(&board, "0", "-");
    let board = str::replace(&board, "1", "*");
    for (i, line) in board.chars().enumerate() {
        if i != 0 && i % width == 0 {
            println!("{}", "");
        }
        print!("{}", line);
    }

    println!("{}", "");
    println!("{}", "");
}

const LEFT_COLUMN_MASK: u128 =                                  0b__100000000__100000000__100000000__100000000__100000000__100000000__100000000__100000000__100000000;
const RIGHT_COLUMN_MASK: u128 =                                 0b__000000001__000000001__000000001__000000001__000000001__000000001__000000001__000000001__000000001;

const VALID_PIECE_POSITION_MASK: u128 =                         0b__111111111__111111111__111111111__111111111__111111111__111111111__111111111__111111111__111111111;
const CANDIDATE_WALL_PLACEMENT_MASK: u128 =                     0b__000000000__011111111__011111111__011111111__011111111__011111111__011111111__011111111__011111111;
const P1_OBJECTIVE_MASK: u128 =                                 0b__111111111__000000000__000000000__000000000__000000000__000000000__000000000__000000000__000000000;
const P2_OBJECTIVE_MASK: u128 =                                 0b__000000000__000000000__000000000__000000000__000000000__000000000__000000000__000000000__111111111;
const P1_STARTING_POS_MASK: u128 =                              0b__000000000__000000000__000000000__000000000__000000000__000000000__000000000__000000000__000010000;
const P2_STARTING_POS_MASK: u128 =                              0b__000010000__000000000__000000000__000000000__000000000__000000000__000000000__000000000__000000000;

const HORIZONTAL_WALL_LEFT_EDGE_TOUCHING_BOARD_MASK: u128 =     0b__000000000__010000000__010000000__010000000__010000000__010000000__010000000__010000000__010000000;
const HORIZONTAL_WALL_RIGHT_EDGE_TOUCHING_BOARD_MASK: u128 =    0b__000000000__000000001__000000001__000000001__000000001__000000001__000000001__000000001__000000001;
const VERTICAL_WALL_TOP_EDGE_TOUCHING_BOARD_MASK: u128 =        0b__000000000__011111111__000000000__000000000__000000000__000000000__000000000__000000000__000000000;
const VERTICAL_WALL_BOTTOM_EDGE_TOUCHING_BOARD_MASK: u128 =     0b__000000000__000000000__000000000__000000000__000000000__000000000__000000000__000000000__011111111;

pub struct GameState {
    pub p1_turn_to_move: bool,
    pub p1_pawn_board: u128,
    pub p2_pawn_board: u128,
    pub p1_num_walls_placed: usize,
    pub p2_num_walls_placed: usize,
    pub vertical_wall_placement_board: u128,
    pub horizontal_wall_placement_board: u128
}

pub struct ValidMoves {
    pub vertical_wall_placement: u128,
    pub horizontal_wall_placement: u128,
    pub pawn_moves: u128
}

struct PathingResult {
    has_path: bool,
    path: u128
}

impl GameState {
    pub fn new() -> GameState {
        GameState {
            p1_turn_to_move: true,
            p1_pawn_board: P1_STARTING_POS_MASK,
            p2_pawn_board: P2_STARTING_POS_MASK,
            p1_num_walls_placed: 0,
            p2_num_walls_placed: 0,
            vertical_wall_placement_board: 0,
            horizontal_wall_placement_board: 0
        }
    }

    pub fn move_pawn(&self, pawn_board: u128) -> GameState {
        let p1_turn_to_move = self.p1_turn_to_move;

        GameState {
            p1_turn_to_move: !p1_turn_to_move,
            p1_pawn_board: if p1_turn_to_move { pawn_board } else { self.p1_pawn_board },
            p2_pawn_board: if !p1_turn_to_move { pawn_board } else { self.p2_pawn_board },
            ..*self
        }
    }

    pub fn place_horizontal_wall(&self, horizontal_wall_placement: u128) -> GameState {
        let p1_turn_to_move = self.p1_turn_to_move;

        GameState {
            p1_turn_to_move: !p1_turn_to_move,
            p1_num_walls_placed: self.p1_num_walls_placed + if p1_turn_to_move { 1 } else { 0 },
            p2_num_walls_placed: self.p2_num_walls_placed + if !p1_turn_to_move { 1 } else { 0 },
            horizontal_wall_placement_board: self.horizontal_wall_placement_board | horizontal_wall_placement,
            ..*self
        }
    }

    pub fn place_vertical_wall(&self, vertical_wall_placement: u128) -> GameState {
        let p1_turn_to_move = self.p1_turn_to_move;

        GameState {
            p1_turn_to_move: !p1_turn_to_move,
            p1_num_walls_placed: self.p1_num_walls_placed + if p1_turn_to_move { 1 } else { 0 },
            p2_num_walls_placed: self.p2_num_walls_placed + if !p1_turn_to_move { 1 } else { 0 },
            vertical_wall_placement_board: self.vertical_wall_placement_board | vertical_wall_placement,
            ..*self
        }
    }

    pub fn get_valid_moves(&self) -> ValidMoves {
        ValidMoves {
            horizontal_wall_placement: self.get_valid_horizontal_wall_placement(),
            vertical_wall_placement: self.get_valid_vertical_wall_placement(),
            pawn_moves: self.get_valid_pawn_moves()
        }
    }

    fn get_valid_pawn_moves(&self) -> u128 {
        let active_player_board = self.get_active_player_board();
        let opposing_player_board = self.get_opposing_player_board();

        let move_up_mask = self.get_move_up_mask();
        let move_right_mask = self.get_move_right_mask();
        let move_down_mask = self.get_move_down_mask();
        let move_left_mask = self.get_move_left_mask();

        let up_move = active_player_board << 9 & move_up_mask;
        let right_move = active_player_board >> 1 & move_right_mask;
        let down_move = active_player_board >> 9 & move_down_mask;
        let left_move = active_player_board << 1 & move_left_mask;

        let valid_moves: u128 = up_move | right_move | down_move | left_move;
        let overlapping_move: u128 = valid_moves & opposing_player_board;

        if overlapping_move == 0 {
            return valid_moves;
        }

        let overlap_up_move = up_move & opposing_player_board;
        let overlap_right_move = right_move & opposing_player_board;
        let overlap_down_move = down_move & opposing_player_board;
        let overlap_left_move = left_move & opposing_player_board;

        let straight_jump_up_move = overlap_up_move << 9 & move_up_mask;
        let straight_jump_right_move = overlap_right_move >> 1 & move_right_mask;
        let straight_jump_down_move = overlap_down_move >> 9 & move_down_mask;
        let straight_jump_left_move = overlap_left_move << 1 & move_left_mask;

        let straight_jump_move = straight_jump_up_move | straight_jump_right_move | straight_jump_down_move | straight_jump_left_move;

        if straight_jump_move != 0 {
            return valid_moves & !opposing_player_board | straight_jump_move;
        }

        let side_jump_moves =
            (
                overlapping_move << 9 & move_up_mask
                | overlapping_move >> 1 & move_right_mask
                | overlapping_move >> 9 & move_down_mask
                | overlapping_move << 1 & move_left_mask
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
        self.vertical_wall_placement_board << 9 | self.vertical_wall_placement_board
    }

    fn get_horizontal_wall_blocks(&self) -> u128 {
        self.horizontal_wall_placement_board << 1 | self.horizontal_wall_placement_board
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
            self.p1_num_walls_placed < 10
        } else {
            self.p2_num_walls_placed < 10
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
        !(self.horizontal_wall_placement_board | self.horizontal_wall_placement_board >> 1 | self.horizontal_wall_placement_board << 1)
        & !self.vertical_wall_placement_board
        & CANDIDATE_WALL_PLACEMENT_MASK
    }

    fn get_candidate_vertical_wall_placement(&self) -> u128 {
        !(self.vertical_wall_placement_board | self.vertical_wall_placement_board >> 9 | self.vertical_wall_placement_board << 9)
        & !self.horizontal_wall_placement_board
        & CANDIDATE_WALL_PLACEMENT_MASK
    }

    fn get_invalid_horizontal_wall_candidates(&self) -> u128 {
        let mut invalid_placements: u128 = 0;
        let mut horizontal_connecting_candidates = self.get_horizontal_connecting_candidates();

        while horizontal_connecting_candidates != 0 {
            let with_removed_candidate = horizontal_connecting_candidates & (horizontal_connecting_candidates - 1);
            let removed_candidate = with_removed_candidate ^ horizontal_connecting_candidates;

            let state_with_candidate = GameState {
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

            let state_with_candidate = GameState {
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
            let up_path = path << 9 & up_mask;
            let right_path = path >> 1 & right_mask;
            let down_path = path >> 9 & down_mask;
            let left_path = path << 1 & left_mask;
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
        !(self.get_horizontal_wall_blocks() << 9) & VALID_PIECE_POSITION_MASK
    }

    fn get_move_right_mask(&self) -> u128 {
        !self.get_vertical_wall_blocks() & VALID_PIECE_POSITION_MASK & !LEFT_COLUMN_MASK
    }

    fn get_move_down_mask(&self) -> u128{
        !self.get_horizontal_wall_blocks() & VALID_PIECE_POSITION_MASK
    }

    fn get_move_left_mask(&self) -> u128 {
        !(self.get_vertical_wall_blocks() << 1) & VALID_PIECE_POSITION_MASK & !RIGHT_COLUMN_MASK
    }

    fn get_horizontal_connecting_candidates(&self) -> u128 {
        let candidate_horizontal_walls = self.get_candidate_horizontal_wall_placement();
        let horizontal_wall_blocks = self.get_horizontal_wall_blocks();
        let vertical_wall_blocks = self.get_vertical_wall_blocks();

        // Compare to existing walls. Wall segment candidates check against locations of possible new connections. These are checked in combinations since the walls may already connected. We don't want to count the same existing connection twice or thrice.
        let middle_touching = candidate_horizontal_walls & (vertical_wall_blocks | vertical_wall_blocks >> 9);
        let left_edge_touching = candidate_horizontal_walls & (vertical_wall_blocks >> 1 | vertical_wall_blocks >> 10 | horizontal_wall_blocks >> 2 | HORIZONTAL_WALL_LEFT_EDGE_TOUCHING_BOARD_MASK);
        let right_edge_touching = candidate_horizontal_walls & (vertical_wall_blocks << 1 | vertical_wall_blocks >> 8 | horizontal_wall_blocks << 1 | HORIZONTAL_WALL_RIGHT_EDGE_TOUCHING_BOARD_MASK);

        (left_edge_touching & middle_touching) | (left_edge_touching & right_edge_touching) | (middle_touching & right_edge_touching)
    }

    fn get_vertical_connecting_candidates(&self) -> u128 {
        let candidate_vertical_walls = self.get_candidate_vertical_wall_placement();
        let vertical_wall_blocks = self.get_vertical_wall_blocks();
        let horizontal_wall_blocks = self.get_horizontal_wall_blocks();

        // Compare to existing walls. Wall segment candidates check against locations of possible new connections. These are checked in combinations since the walls may already connected. We don't want to count the same existing connection twice or thrice.
        let middle_touching = candidate_vertical_walls & (horizontal_wall_blocks | horizontal_wall_blocks >> 1);
        let top_edge_touching = candidate_vertical_walls & (horizontal_wall_blocks >> 9 | horizontal_wall_blocks >> 10 | vertical_wall_blocks >> 18 | VERTICAL_WALL_TOP_EDGE_TOUCHING_BOARD_MASK);
        let bottom_edge_touching = candidate_vertical_walls & (horizontal_wall_blocks << 8 | horizontal_wall_blocks << 9 | vertical_wall_blocks << 9 | VERTICAL_WALL_BOTTOM_EDGE_TOUCHING_BOARD_MASK);

        (top_edge_touching & middle_touching) | (top_edge_touching & bottom_edge_touching) | (middle_touching & bottom_edge_touching)
    }
}
