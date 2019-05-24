mod engine;

// https://github.com/dgrunwald/rust-cpython
#[macro_use] extern crate cpython;

use cpython::{PyResult, Python};

// add bindings to the generated python module
// N.B: names: "librust2py" must be the name of the `.so` or `.pyd` file
py_module_initializer!(librust2py, initlibrust2py, PyInit_librust2py, |py, m| {
    m.add(py, "__doc__", "This module is implemented in Rust.")?;
    m.add(py, "new_game", py_fn!(py, new_game()))?;
    Ok(())
});

type GameStateTuple = (
    bool,         // p1_turn_to_move
    Vec<f32>,     // p1_pawn_board
    Vec<f32>,     // p2_pawn_board
    usize,        // p1_num_walls_placed
    usize,        // p2_num_walls_placed
    Vec<f32>,     // vertical_wall_placement_board
    Vec<f32>,     // horizontal_wall_placement_board
);

// rust-cpython aware function. All of our python interface could be
// declared in a separate module.
// Note that the py_fn!() macro automatically converts the arguments from
// Python objects to Rust values; and the Rust return value back into a Python object.
fn new_game(_: Python) -> PyResult<GameStateTuple> {
    let game_state = engine::GameState::new();
    let game_state_tuple = map_game_state_to_tuple(game_state);

    Ok(game_state_tuple)
}


fn map_game_state_to_tuple(game_state: engine::GameState) -> GameStateTuple {
    (
        game_state.p1_turn_to_move,
        map_board_to_vec(game_state.p1_pawn_board).to_vec(),
        map_board_to_vec(game_state.p2_pawn_board).to_vec(),
        game_state.p1_num_walls_placed,
        game_state.p2_num_walls_placed,
        map_board_to_vec(game_state.vertical_wall_placement_board).to_vec(),
        map_board_to_vec(game_state.horizontal_wall_placement_board).to_vec(),
    )
}

fn map_board_to_vec(mut board: u128) -> [f32; 81] {
    let mut result:[f32; 81] = [0.0; 81];
    while board != 0 {
        let board_without_first_bit = board & (board - 1);
        let removed_bit = board ^ board_without_first_bit;
        let removed_bit_idx = single_bit_index(removed_bit);
        result[removed_bit_idx] = 1.0;
        board = board_without_first_bit;
    }

    result
}

fn single_bit_index(mut bit: u128) -> usize {
    let mut n = 0;

    if bit >> 64 != 0 {
        n += 64;
        bit >>= 64;
    }
    if bit >> 32 != 0 {
        n += 32;
        bit >>= 32;
    }
    if bit >> 16 != 0 {
        n += 16;
        bit >>= 16;
    }
    if bit >> 8 != 0 {
        n += 8;
        bit >>= 8;
    }
    if bit >> 4 != 0 {
        n += 4;
        bit >>= 4;
    }
    if bit >> 2 != 0 {
        n += 2;
        bit >>= 2;
    }
    if bit >> 1 != 0 {
        n += 1;
    }

    return n;
}
