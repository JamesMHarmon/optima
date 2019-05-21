fn main() {
    let piece_board: u128 = 0b__000000010__000000000__;
    let vertical_wall_placement_board: u128 = 0b__000000000__000100010__000000000__001000001__000000000__;
    let horizontal_wall_placement_board: u128 = 0b__000000000__000010100__000000000_000100001__000000000__000000000__;

    let game_state = GameState {
        p1_piece_board: piece_board,
        p2_piece_board: piece_board,
        vertical_wall_placement_board: vertical_wall_placement_board,
        horizontal_wall_placement_board: horizontal_wall_placement_board,
    };

    println!("{}", "Vertical Walls:");
    print_wall_board(game_state.get_vertical_wall_placement());

    println!("{}", "Horizontal Walls:");
    print_wall_board(game_state.get_horizontal_wall_placement());

    println!("{}", "Starting Pos:");
    print_piece_board(game_state.p1_piece_board);

    println!("{}", "get_candidate_horizontal_wall_placement:");
    print_wall_board(game_state.get_candidate_horizontal_wall_placement());
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

const CANDIDATE_HORIZONTAL_WALL_PLACEMENT_MASK: u128 = 0b__000000000__011111111__011111111__011111111__011111111__011111111__011111111__011111111__011111111;


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
        let horizontal_wall_placement = self.get_horizontal_wall_placement();
        (self.p1_piece_board & horizontal_wall_placement) | (self.p1_piece_board & TOP_ROW_MASK) == 0
    }

    fn can_move_right(&self) -> bool {
        let vertical_wall_placement = self.get_vertical_wall_placement();
        let p1_piece_move = self.p1_piece_board >> 1;
        (p1_piece_move & vertical_wall_placement) | (self.p1_piece_board & RIGHT_COLUMN_MASK) == 0
    }

    fn can_move_down(&self) -> bool {
        let horizontal_wall_placement = self.get_horizontal_wall_placement();
        let p1_piece_move = self.p1_piece_board >> 9;
        (p1_piece_move & horizontal_wall_placement) | (self.p1_piece_board & BOTTOM_ROW_MASK) == 0
    }

    fn can_move_left(&self) -> bool {
        let vertical_wall_placement = self.get_vertical_wall_placement();
        (self.p1_piece_board & vertical_wall_placement) | (self.p1_piece_board & LEFT_COLUMN_MASK) == 0
    }

    fn get_vertical_wall_placement(&self) -> u128 {
        self.vertical_wall_placement_board << 9 | self.vertical_wall_placement_board
    }

    fn get_horizontal_wall_placement(&self) -> u128 {
        self.horizontal_wall_placement_board << 1 | self.horizontal_wall_placement_board
    }

    fn get_candidate_horizontal_wall_placement(&self) -> u128  {
        !(self.horizontal_wall_placement_board | self.horizontal_wall_placement_board >> 1 | self.horizontal_wall_placement_board << 1)
        & !self.vertical_wall_placement_board
        & CANDIDATE_HORIZONTAL_WALL_PLACEMENT_MASK
    }
}
