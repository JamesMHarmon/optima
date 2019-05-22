fn main() {
    let game_state = GameState {
        p1_piece_board:                     0b__000000000__000000000__000000000__000000000__000000000__000000000__000000000__000000000__000010000__,
        p2_piece_board:                     0b__000000000__000000000__001000000__000000000__000000000__000000000__000000000__000000000__000000000__,
        vertical_wall_placement_board:      0b__000000000__000000000__000000000__000000000__000000000__000010010__000000000__001000001__000000000__,
        horizontal_wall_placement_board:    0b__000000000__000000000__000000000__000000000__000010100__000000000__000100001__000000000__000101010__,
    };

    println!("{}", "Vertical Walls:");
    print_board(game_state.get_vertical_wall_blocks());

    println!("{}", "Horizontal Walls:");
    print_board(game_state.get_horizontal_wall_blocks());

    println!("{}", "Starting Pos:");
    print_board(game_state.p1_piece_board);

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
const CANDIDATE_HORIZONTAL_WALL_PLACEMENT_MASK: u128 =          0b__000000000__011111111__011111111__011111111__011111111__011111111__011111111__011111111__011111111;
const P1_OBJECTIVE_MASK: u128 =                                 0b__111111111__000000000__000000000__000000000__000000000__000000000__000000000__000000000__000000000;
const P2_OBJECTIVE_MASK: u128 =                                 0b__000000000__000000000__000000000__000000000__000000000__000000000__000000000__000000000__111111111;

const HORIZONTAL_WALL_LEFT_EDGE_TOUCHING_BOARD_MASK: u128 =     0b__000000000__010000000__010000000__010000000__010000000__010000000__010000000__010000000__010000000;
const HORIZONTAL_WALL_RIGHT_EDGE_TOUCHING_BOARD_MASK: u128 =    0b__000000000__000000001__000000001__000000001__000000001__000000001__000000001__000000001__000000001;


struct GameState {
    p1_piece_board: u128,
    p2_piece_board: u128,
    vertical_wall_placement_board: u128,
    horizontal_wall_placement_board: u128
}

struct PathingResult {
    hasPath: bool,
    path: u128
}

impl GameState {
    fn move_up(&self) -> GameState {
        GameState {
            p1_piece_board: self.p1_piece_board << 9,
            ..*self
        }
    }

    fn move_right(&self) -> GameState {
        GameState {
            p1_piece_board: self.p1_piece_board >> 1,
            ..*self
        }
    }

    fn move_down(&self) -> GameState {
        GameState {
            p1_piece_board: self.p1_piece_board >> 9,
            ..*self
        }
    }

    fn move_left(&self) -> GameState {
        GameState {
            p1_piece_board: self.p1_piece_board << 1,
            ..*self
        }
    }

    fn can_move_up(&self) -> bool {
        let horizontal_wall_blocks = self.get_horizontal_wall_blocks();
        (self.p1_piece_board & horizontal_wall_blocks) | (self.p1_piece_board & TOP_ROW_MASK) == 0
    }

    fn can_move_right(&self) -> bool {
        let vertical_wall_blocks = self.get_vertical_wall_blocks();
        let p1_piece_move = self.p1_piece_board >> 1;
        (p1_piece_move & vertical_wall_blocks) | (self.p1_piece_board & RIGHT_COLUMN_MASK) == 0
    }

    fn can_move_down(&self) -> bool {
        let horizontal_wall_blocks = self.get_horizontal_wall_blocks();
        let p1_piece_move = self.p1_piece_board >> 9;
        (p1_piece_move & horizontal_wall_blocks) | (self.p1_piece_board & BOTTOM_ROW_MASK) == 0
    }

    fn can_move_left(&self) -> bool {
        let vertical_wall_blocks = self.get_vertical_wall_blocks();
        (self.p1_piece_board & vertical_wall_blocks) | (self.p1_piece_board & LEFT_COLUMN_MASK) == 0
    }

    fn get_vertical_wall_blocks(&self) -> u128 {
        self.vertical_wall_placement_board << 9 | self.vertical_wall_placement_board
    }

    fn get_horizontal_wall_blocks(&self) -> u128 {
        self.horizontal_wall_placement_board << 1 | self.horizontal_wall_placement_board
    }

    pub fn get_valid_horizontal_wall_positions(&self) -> u128 {
        let candidate_horizontal_wall_placement = self.get_candidate_horizontal_wall_placement();
        let invalid_horizontal_candidates = self.get_invalid_horizontal_wall_candidates();
        candidate_horizontal_wall_placement & !invalid_horizontal_candidates
    }

    fn get_candidate_horizontal_wall_placement(&self) -> u128 {
        !(self.horizontal_wall_placement_board | self.horizontal_wall_placement_board >> 1 | self.horizontal_wall_placement_board << 1)
        & !self.vertical_wall_placement_board
        & CANDIDATE_HORIZONTAL_WALL_PLACEMENT_MASK
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

    // @TODO: Add possible moves for jumping over the opponent's piece
    fn get_jump_moves() {

    }

    fn players_have_path(&self) -> bool {
        let p1_path_result = self.find_path(self.p1_piece_board, P1_OBJECTIVE_MASK);

        if !p1_path_result.hasPath {
            return false;
        }

        // If the pathing result generated from checking for p1's path overlaps the p2 pawn, we can start where that path left off to
        // save a few cycles.
        let p2_path_start = if p1_path_result.path & self.p2_piece_board != 0 { p1_path_result.path } else { self.p2_piece_board };

        self.find_path(p2_path_start, P2_OBJECTIVE_MASK).hasPath
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
                return PathingResult { hasPath: true, path: updated_path };
            }

            // Check if any progress was made, if there is no progress from the last iteration then we are stuck.
            if updated_path == path {
                return PathingResult { hasPath: false, path: updated_path };
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
}
