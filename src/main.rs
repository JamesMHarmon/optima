use std::time::Instant;

fn main() {
    let game_state = GameState {
        p1_turn_to_move: true,
        p1_remaining_walls: 10,
        p2_remaining_walls: 10,
        p1_pawn_board:                     0b__000000000__000000000__000000000__000000000__000000000__000000000__000000000__000000000__000010000__,
        p2_pawn_board:                     0b__000000000__000000000__001000000__000000000__000000000__000000000__000000000__000000000__000000000__,
        vertical_wall_placement_board:     0b__000000000__000000000__000000000__000000000__000000000__000010010__000000000__001000001__000000000__,
        horizontal_wall_placement_board:   0b__000000000__000000000__000000000__000000000__000010100__000000000__000100001__000000000__000101010__,
        // vertical_wall_placement_board:     0b__000000000__000000000__000000000__000000000__000000000__000000000__000000000__000000000__000000000__,
        // horizontal_wall_placement_board:   0b__000000000__000000000__000000000__000000000__000000000__000000000__000000000__000000000__000000000__,
    };

    let mut valid_moves: ValidMoves = game_state.get_valid_moves();

    let now = Instant::now();
    for _ in 1..1_000_000 {
        valid_moves = game_state.get_valid_moves()
    }

    let elapsed = now.elapsed().as_millis();

    println!("{}", "Vertical Walls:");
    print_board(game_state.get_vertical_wall_blocks());

    println!("{}", "Horizontal Walls:");
    print_board(game_state.get_horizontal_wall_blocks());

    println!("{}", "Starting Pos:");
    print_board(game_state.p1_pawn_board);

    println!("{}", "get_candidate_horizontal_wall_placement:");
    print_board(game_state.get_candidate_horizontal_wall_placement());

    println!("{}", "has_path");
    println!("{}", game_state.players_have_path());

    println!("{}", "get_horizontal_connecting_candidates");
    print_board(game_state.get_horizontal_connecting_candidates());

    println!("{}", "get_invalid_horizontal_wall_candidates");
    print_board(game_state.get_invalid_horizontal_wall_candidates());

    println!("{}", "get_valid_horizontal_wall_positions");
    print_board(game_state.get_valid_horizontal_wall_positions());

    println!("Valid Moves: {}", valid_moves.can_move_up);

    println!("Elapsed Time: {}", elapsed);
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

const TOP_ROW_MASK: u128 =                                      0b__111111111__000000000__000000000__000000000__000000000__000000000__000000000__000000000__000000000;
const BOTTOM_ROW_MASK: u128 =                                   0b__000000000__000000000__000000000__000000000__000000000__000000000__000000000__000000000__111111111;
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
    pub p1_remaining_walls: usize,
    pub p2_remaining_walls: usize,
    pub vertical_wall_placement_board: u128,
    pub horizontal_wall_placement_board: u128
}

pub struct ValidMoves {
    pub vertical_wall_placement: u128,
    pub horizontal_wall_placement: u128,
    pub can_move_up: bool,
    pub can_move_right: bool,
    pub can_move_down: bool,
    pub can_move_left: bool,
    pub can_jump_up: bool,
    pub can_jump_upper_right: bool,
    pub can_jump_right: bool,
    pub can_jump_lower_right: bool,
    pub can_jump_down: bool,
    pub can_jump_lower_left: bool,
    pub can_jump_left: bool,
    pub can_jump_upper_left: bool,
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
            p1_remaining_walls: 10,
            p2_remaining_walls: 10,
            vertical_wall_placement_board: 0,
            horizontal_wall_placement_board: 0
        }
    }

    pub fn move_up(&self) -> GameState {
        GameState {
            p1_turn_to_move: !self.p1_turn_to_move,
            p1_pawn_board: self.p1_pawn_board << 9,
            ..*self
        }
    }

    pub fn move_right(&self) -> GameState {
        GameState {
            p1_turn_to_move: !self.p1_turn_to_move,
            p1_pawn_board: self.p1_pawn_board >> 1,
            ..*self
        }
    }

    pub fn move_down(&self) -> GameState {
        GameState {
            p1_turn_to_move: !self.p1_turn_to_move,
            p1_pawn_board: self.p1_pawn_board >> 9,
            ..*self
        }
    }

    pub fn move_left(&self) -> GameState {
        GameState {
            p1_turn_to_move: !self.p1_turn_to_move,
            p1_pawn_board: self.p1_pawn_board << 1,
            ..*self
        }
    }

    pub fn get_valid_moves(&self) -> ValidMoves {
        let valid_jumps = self.can_jump();

        ValidMoves {
            horizontal_wall_placement: self.get_valid_horizontal_wall_placement(),
            vertical_wall_placement: self.get_valid_vertical_wall_placement(),
            can_move_up: self.can_move_up(),
            can_move_right: self.can_move_right(),
            can_move_down: self.can_move_down(),
            can_move_left: self.can_move_left(),
            can_jump_up: valid_jumps.0,
            can_jump_upper_right: valid_jumps.1,
            can_jump_right: valid_jumps.2,
            can_jump_lower_right: valid_jumps.3,
            can_jump_down: valid_jumps.4,
            can_jump_lower_left: valid_jumps.5,
            can_jump_left: valid_jumps.6,
            can_jump_upper_left: valid_jumps.7,
        }
    }

    fn can_move_up(&self) -> bool {
        let horizontal_wall_blocks = self.get_horizontal_wall_blocks();
        let active_player_board = self.get_active_player_board();
        (active_player_board & horizontal_wall_blocks) | (active_player_board & TOP_ROW_MASK) == 0
    }

    fn can_move_right(&self) -> bool {
        let vertical_wall_blocks = self.get_vertical_wall_blocks();
        let active_player_board = self.get_active_player_board();
        let pawn_move = active_player_board >> 1;
        (pawn_move & vertical_wall_blocks) | (active_player_board & RIGHT_COLUMN_MASK) == 0
    }

    fn can_move_down(&self) -> bool {
        let horizontal_wall_blocks = self.get_horizontal_wall_blocks();
        let active_player_board = self.get_active_player_board();
        let pawn_move = active_player_board >> 9;
        (pawn_move & horizontal_wall_blocks) | (active_player_board & BOTTOM_ROW_MASK) == 0
    }

    fn can_move_left(&self) -> bool {
        let vertical_wall_blocks = self.get_vertical_wall_blocks();
        let active_player_board = self.get_active_player_board();
        (active_player_board & vertical_wall_blocks) | (active_player_board & LEFT_COLUMN_MASK) == 0
    }

    fn can_jump(&self) -> (bool, bool, bool, bool, bool, bool, bool, bool) {
        let active_player = self.get_active_player_board();
        let opponent = self.get_opposing_player_board();

        let opponent_is_up = active_player << 9 & opponent != 0 && self.can_move_up();
        let opponent_is_right = active_player >> 1 & opponent != 0 && self.can_move_right();
        let opponent_is_down = active_player >> 9 & opponent != 0 && self.can_move_down();
        let opponent_is_left = active_player << 1 & opponent != 0 && self.can_move_left();

        let can_jump_up_up = opponent_is_up && self.move_up().can_move_up();
        let can_jump_straight_right = opponent_is_right && self.move_right().can_move_right();
        let can_jump_straight_down = opponent_is_down && self.move_down().can_move_down();
        let can_jump_straight_left = opponent_is_left && self.move_left().can_move_left();

        let can_jump_up_right = opponent_is_up && !can_jump_up_up && self.move_up().can_move_right();
        let can_jump_up_left = opponent_is_up && !can_jump_up_up && self.move_up().can_move_left();

        let can_jump_right_up = opponent_is_right && !can_jump_straight_right && self.move_right().can_move_up();
        let can_jump_right_down = opponent_is_right && !can_jump_straight_right && self.move_right().can_move_down();

        let can_jump_down_right = opponent_is_down && !can_jump_straight_down && self.move_down().can_move_right();
        let can_jump_down_left = opponent_is_down && !can_jump_straight_down && self.move_down().can_move_left();

        let can_jump_left_up = opponent_is_left && !can_jump_straight_left && self.move_left().can_move_up();
        let can_jump_left_down = opponent_is_left && !can_jump_straight_left && self.move_left().can_move_down();

        (
            can_jump_up_up,
            can_jump_up_right || can_jump_right_up,
            can_jump_straight_right,
            can_jump_right_down || can_jump_down_right,
            can_jump_straight_down,
            can_jump_down_left || can_jump_left_down,
            can_jump_straight_left,
            can_jump_left_up || can_jump_up_left
        )
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
        if self.active_player_has_walls() {
            self.get_valid_horizontal_wall_positions()
        } else {
            0
        }
    }

    fn get_valid_vertical_wall_placement(&self) -> u128 {
        if self.active_player_has_walls() {
            self.get_valid_vertical_wall_positions()
        } else {
            0
        }
    }

    fn active_player_has_walls(&self) -> bool {
        if self.p1_turn_to_move {
            self.p1_remaining_walls > 0
        } else {
            self.p2_remaining_walls > 0
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
        let mut potential_invalid_placements: Vec<u128> = Vec::new();
        let mut horizontal_connecting_candidates = self.get_horizontal_connecting_candidates();

        while horizontal_connecting_candidates != 0 {
            let with_removed_candidate = horizontal_connecting_candidates & (horizontal_connecting_candidates - 1);
            let removed_candidate = with_removed_candidate ^ horizontal_connecting_candidates;
            potential_invalid_placements.push(removed_candidate);
            horizontal_connecting_candidates = with_removed_candidate;
        }

        self.recursive_get_invalid_horizontal_candidate(&potential_invalid_placements[..])
    }

    fn get_invalid_vertical_wall_candidates(&self) -> u128 {
        let mut potential_invalid_placements: Vec<u128> = Vec::new();
        let mut vertical_connecting_candidates = self.get_vertical_connecting_candidates();

        while vertical_connecting_candidates != 0 {
            let with_removed_candidate = vertical_connecting_candidates & (vertical_connecting_candidates - 1);
            let removed_candidate = with_removed_candidate ^ vertical_connecting_candidates;
            potential_invalid_placements.push(removed_candidate);
            vertical_connecting_candidates = with_removed_candidate;
        }

        self.recursive_get_invalid_vertical_candidate(&potential_invalid_placements[..])
    }

    fn recursive_get_invalid_horizontal_candidate(&self, potential_invalid_placements: &[u128]) -> u128 {
        let number_of_potential_placements = potential_invalid_placements.len();
        if number_of_potential_placements == 0 {
            return 0;
        }

        let mut potential_invalid_placements_mask: u128 = 0;
        for potential_invalid_placement in potential_invalid_placements {
            potential_invalid_placements_mask |= potential_invalid_placement;
        }

        let game_state_with_walls = GameState {
            horizontal_wall_placement_board: self.horizontal_wall_placement_board | potential_invalid_placements_mask,
            ..*self
        };

        if game_state_with_walls.players_have_path() {
            return 0;
        }

        let number_of_potential_placements = potential_invalid_placements.len();

        if number_of_potential_placements == 1 {
            return potential_invalid_placements[0];
        }

        let half_of_length = number_of_potential_placements / 2;

        self.recursive_get_invalid_horizontal_candidate(&potential_invalid_placements[..half_of_length])
        | self.recursive_get_invalid_horizontal_candidate(&potential_invalid_placements[half_of_length..])
    }

    fn recursive_get_invalid_vertical_candidate(&self, potential_invalid_placements: &[u128]) -> u128 {
        let number_of_potential_placements = potential_invalid_placements.len();
        if number_of_potential_placements == 0 {
            return 0;
        }

        let mut potential_invalid_placements_mask: u128 = 0;
        for potential_invalid_placement in potential_invalid_placements {
            potential_invalid_placements_mask |= potential_invalid_placement;
        }

        let game_state_with_walls = GameState {
            vertical_wall_placement_board: self.vertical_wall_placement_board | potential_invalid_placements_mask,
            ..*self
        };

        if game_state_with_walls.players_have_path() {
            return 0;
        }

        let number_of_potential_placements = potential_invalid_placements.len();

        if number_of_potential_placements == 1 {
            return potential_invalid_placements[0];
        }

        let half_of_length = number_of_potential_placements / 2;

        self.recursive_get_invalid_vertical_candidate(&potential_invalid_placements[..half_of_length])
        | self.recursive_get_invalid_vertical_candidate(&potential_invalid_placements[half_of_length..])
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
        let horizontal_wall_blocks = self.get_horizontal_wall_blocks();
        let vertical_wall_blocks = self.get_vertical_wall_blocks();
        let up_mask = !(horizontal_wall_blocks << 9) & VALID_PIECE_POSITION_MASK;
        let right_mask = !vertical_wall_blocks & VALID_PIECE_POSITION_MASK & !LEFT_COLUMN_MASK;
        let down_mask = !horizontal_wall_blocks & VALID_PIECE_POSITION_MASK;
        let left_mask = !(vertical_wall_blocks << 1) & VALID_PIECE_POSITION_MASK & !RIGHT_COLUMN_MASK;

        let mut path = start;

        loop {
            // MOVE UP & DOWN & LEFT & RIGHT
            let up_path = (path << 9) & up_mask;
            let right_path = (path >> 1) & right_mask;
            let down_path = (path >> 9) & down_mask;
            let left_path = (path << 1) & left_mask;
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
