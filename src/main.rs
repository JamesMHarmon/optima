extern crate quoridor;

fn main() {
    let game_state = quoridor::engine::GameState {
        p1_turn_to_move: true,
        p1_num_walls_placed: 0,
        p2_num_walls_placed: 0,
        p1_pawn_board:                     0b__000000000__000010000__000000000__000000000__000000000__000000000__000000000__000000000__000000000__,
        p2_pawn_board:                     0b__000000000__001000000__000000000__000000000__000000000__000000000__000000000__000010000__000000000__,
        vertical_wall_placement_board:     0b__000000000__000000000__000000000__000000000__000000000__000010010__000000000__001000001__000000000__,
        horizontal_wall_placement_board:   0b__000000000__000000000__000000000__000000000__000010100__000000000__000100001__000000000__000101010__,
    };

    let game_state = quoridor::engine::GameState::new();
    let game_state = game_state.move_pawn(game_state.p1_pawn_board << 9);
    let game_state = game_state.place_horizontal_wall(0b000010000_000000000);
    let game_state = game_state.move_pawn(game_state.p1_pawn_board << 1);

    // println!("{}", "Vertical Walls:");
    // print_board(game_state.get_vertical_wall_blocks());

    // println!("Horizontal Walls:");
    // print_board(game_state.get_horizontal_wall_blocks());

    println!("Pawn Positions:");
    print_board(game_state.p1_pawn_board | game_state.p2_pawn_board);

    // println!("get_candidate_horizontal_wall_placement:");
    // print_board(game_state.get_candidate_horizontal_wall_placement());

    // println!("has_path");
    // println!("{}", game_state.players_have_path());

    // println!("get_horizontal_connecting_candidates");
    // print_board(game_state.get_horizontal_connecting_candidates());

    // println!("get_invalid_horizontal_wall_candidates");
    // print_board(game_state.get_invalid_horizontal_wall_candidates());

    // println!("get_valid_horizontal_wall_positions");
    // print_board(game_state.get_valid_horizontal_wall_positions());

    println!("Valid Moves");
    println!("{:?}", game_state.get_valid_moves());
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
