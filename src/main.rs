fn main() {
    let game_state = GameState {
        p1_piece_board: 0b__000000000__000000000__000000000__000000000__000000000__000010000__000000000__000000000__000000000__,
        p2_piece_board: 0b__000000010__000000000__,
        vertical_wall_placement_board: 0b__000000000__000000000__000000000__000000000__000000000__,
        horizontal_wall_placement_board: 0b__000000000__000010100__000000000_000100001__000000000__000101010__,
    };

    println!("{}", "Vertical Walls:");
    print_wall_board(game_state.get_vertical_wall_blocks());

    println!("{}", "Horizontal Walls:");
    print_wall_board(game_state.get_horizontal_wall_blocks());

    println!("{}", "Starting Pos:");
    print_piece_board(game_state.p1_piece_board);

    println!("{}", "get_candidate_horizontal_wall_placement:");
    print_wall_board(game_state.get_candidate_horizontal_wall_placement());

    println!("{}", "has_path");
    println!("{}", game_state.p1_has_path());

    println!("{}", "get_horizontal_connecting_candidates");
    print_wall_board(game_state.get_horizontal_connecting_candidates());
}

fn print_piece_board(piece_board: u128) {
    let board = format!("{:081b}", piece_board);
    print_board(&board, 9);
}

fn print_wall_board(wall_board: u128) {
    let board = format!("{:081b}", wall_board);
    print_board(&board, 9);
}

fn print_board(board: &str, width: usize) {
    let board = str::replace(board, "0", "-");
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


const TOP_ROW_MASK: u128 =      0b__111111111__000000000__000000000__000000000__000000000__000000000__000000000__000000000__000000000;
const BOTTOM_ROW_MASK: u128 =   0b__000000000__000000000__000000000__000000000__000000000__000000000__000000000__000000000__111111111;
const LEFT_COLUMN_MASK: u128 =  0b__100000000__100000000__100000000__100000000__100000000__100000000__100000000__100000000__100000000;
const RIGHT_COLUMN_MASK: u128 = 0b__000000001__000000001__000000001__000000001__000000001__000000001__000000001__000000001__000000001;

const VALID_PIECE_POSITION_MASK: u128 = 0b__111111111__111111111__111111111__111111111__111111111__111111111__111111111__111111111__111111111;
const CANDIDATE_HORIZONTAL_WALL_PLACEMENT_MASK: u128 = 0b__000000000__011111111__011111111__011111111__011111111__011111111__011111111__011111111__011111111;
const P1_OBJECTIVE_MASK: u128 = 0b__111111111__000000000__000000000__000000000__000000000__000000000__000000000__000000000__000000000;

const HORIZONTAL_WALL_LEFT_EDGE_TOUCHING_BOARD_MASK: u128 = 0b__000000000__010000000__010000000__010000000__010000000__010000000__010000000__010000000__010000000;
const HORIZONTAL_WALL_RIGHT_EDGE_TOUCHING_BOARD_MASK: u128 = 0b__000000000__000000001__000000001__000000001__000000001__000000001__000000001__000000001__000000001;


struct GameState {
    p1_piece_board: u128,
    p2_piece_board: u128,
    vertical_wall_placement_board: u128,
    horizontal_wall_placement_board: u128
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

    fn get_candidate_horizontal_wall_placement(&self) -> u128 {
        !(self.horizontal_wall_placement_board | self.horizontal_wall_placement_board >> 1 | self.horizontal_wall_placement_board << 1)
        & !self.vertical_wall_placement_board
        & CANDIDATE_HORIZONTAL_WALL_PLACEMENT_MASK
    }

    // @TODO: Add possible moves for jumping over the opponent's piece

    fn p1_has_path(&self) -> bool {
        let horizontal_wall_blocks = self.get_horizontal_wall_blocks();
        let vertical_wall_blocks = self.get_vertical_wall_blocks();
        let up_mask = !(horizontal_wall_blocks << 9) & VALID_PIECE_POSITION_MASK;
        let right_mask = !vertical_wall_blocks & VALID_PIECE_POSITION_MASK & !LEFT_COLUMN_MASK;
        let down_mask = !horizontal_wall_blocks & VALID_PIECE_POSITION_MASK;
        let left_mask = !(vertical_wall_blocks << 1) & VALID_PIECE_POSITION_MASK & !RIGHT_COLUMN_MASK;

        let mut movements = self.p1_piece_board;

        loop {
            println!("{}", "movements:");
            print_piece_board(movements);

            // MOVE UP & DOWN & LEFT & RIGHT
            let up_movements = (movements << 9) & up_mask;
            let right_movements = (movements >> 1) & right_mask;
            let down_movements = (movements >> 9) & down_mask;
            let left_movements = (movements << 1) & left_mask;
            let updated_movements = up_movements | right_movements | down_movements | left_movements | movements;

            // Check if the objective is reachable
            if P1_OBJECTIVE_MASK & updated_movements != 0 {
                return true;
            }

            // Check if any progress was made, if there is no progress from the last iteration then we are stuck.
            if updated_movements == movements {
                return false;
            }

            movements = updated_movements;
        }
    }

    // Positions where a candidate horizontal wall is touching the edge of the board.
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
