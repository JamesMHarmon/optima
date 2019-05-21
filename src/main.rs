fn main() {
    let board: u128 = 0b001_000_000;

    println!("{}", "Starting Pos:");
    print_board(board);
    
    let board = move_piece_left(board);
    
    println!("{}", "Move Left:");
    print_board(board);
    
    let board = move_piece_up(board);
    
    println!("{}", "Move Up:");
    print_board(board);
    
    let board = move_piece_right(board);
    
    println!("{}", "Move Right:");
    print_board(board);
    
    let board = move_piece_down(board);
    
    println!("{}", "Move Down:");
    print_board(board);
}

fn print_board(board: u128) {
    let board: String = format!("{:081b}", board);
    for (i, line) in board.chars().enumerate() {
        if i != 0 && i % 9 == 0 {
            println!("{}", "");
        }
        print!("{}", line);
    }

    println!("{}", "");
    println!("{}", "");
}

fn move_piece_left(board: u128) -> u128 {
    board << 1
}

fn move_piece_up(board: u128) -> u128 {
    board << 9
}

fn move_piece_right(board: u128) -> u128 {
    board >> 1
}

fn move_piece_down(board: u128) -> u128 {
    board >> 9
}
