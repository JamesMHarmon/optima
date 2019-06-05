pub mod engine;
pub mod mcts;
pub mod analysis;
pub mod quoridor;

// https://github.com/dgrunwald/rust-cpython
// #[macro_use] extern crate cpython;

// use pyo3::prelude::*;
// use pyo3::wrap_pyfunction;
// use pyo3::types::{PyTuple,PyList,PyAny};
// use pyo3::{FromPy, FromPyObject};

// #[pymodule]
// fn quoridor_cpython(py: Python, m: &PyModule) -> PyResult<()> {
//     m.add_wrapped(wrap_pyfunction!(new_game))?;
//     m.add_wrapped(wrap_pyfunction!(move_pawn))?;
//     Ok(())
// }

// type GameStateTuple = (
//     bool,         // p1_turn_to_move
//     Vec<f32>,     // p1_pawn_board
//     Vec<f32>,     // p2_pawn_board
//     usize,        // p1_num_walls_placed
//     usize,        // p2_num_walls_placed
//     Vec<f32>,     // vertical_wall_placement_board
//     Vec<f32>,     // horizontal_wall_placement_board
// );

// // rust-cpython aware function. All of our python interface could be
// // declared in a separate module.
// // Note that the py_fn!() macro automatically converts the arguments from
// // Python objects to Rust values; and the Rust return value back into a Python object.
// #[pyfunction]
// fn new_game() -> PyResult<GameStateTuple> {
//     let game_state = engine::GameState::new();
//     let game_state_tuple = map_game_state_to_tuple(&game_state);

//     Ok(game_state_tuple)
// }

// #[pyfunction]
// fn move_pawn(game_state_tuple: &PyTuple, pawn_board: &PyList) -> PyResult<GameStateTuple> {
//     panic!()
//     // let game_state = map_tuple_to_game_state(game_state_tuple);
//     // let game_state = game_state.move_pawn(map_vec_to_board(pawn_board));
//     // let game_state_tuple = map_game_state_to_tuple(&game_state);

//     // Ok(game_state_tuple)
// }

// fn map_tuple_to_game_state(game_state_tuple: &PyTuple) -> engine::GameState {
//     panic!()
//     // let test: f32 = FromPyObject::extract(game_state_tuple.get_item(0)).unwrap();
//     // engine::GameState {
//     //     p1_turn_to_move: (FromPyObject::extract(game_state_tuple.get_item(0)).unwrap()),
//     //     p1_pawn_board: map_vec_to_board(FromPyObject::extract(FromPyObject::extract(game_state_tuple.get_item(1)).unwrap() as &PyList)),
//     //     p2_pawn_board: map_vec_to_board(FromPyObject::extract(game_state_tuple.get_item(2)).unwrap()),
//     //     p1_num_walls_placed: FromPyObject::extract(game_state_tuple.get_item(3)).unwrap(),
//     //     p2_num_walls_placed: FromPyObject::extract(game_state_tuple.get_item(4)).unwrap(),
//     //     vertical_wall_placement_board: map_vec_to_board(FromPyObject::extract(game_state_tuple.get_item(5)).unwrap()),
//     //     horizontal_wall_placement_board: map_vec_to_board(FromPyObject::extract(game_state_tuple.get_item(6)).unwrap()),
//     // }
// }

// fn map_game_state_to_tuple(game_state: &engine::GameState) -> GameStateTuple {
//     (
//         game_state.p1_turn_to_move,
//         map_board_to_vec(game_state.p1_pawn_board).to_vec(),
//         map_board_to_vec(game_state.p2_pawn_board).to_vec(),
//         game_state.p1_num_walls_placed,
//         game_state.p2_num_walls_placed,
//         map_board_to_vec(game_state.vertical_wall_placement_board).to_vec(),
//         map_board_to_vec(game_state.horizontal_wall_placement_board).to_vec(),
//     )
// }

// fn map_vec_to_board(board: &Vec<f32>) -> u128 {
//     let mut bit_board: u128 = 0;
//     for (i, bits) in board.iter().enumerate() {
//         let mask: u128 = 1;
//         bit_board |= mask << i;
//     }
//     bit_board
// }

// fn map_board_to_vec(mut board: u128) -> [f32; 81] {
//     let mut result:[f32; 81] = [0.0; 81];
//     while board != 0 {
//         let board_without_first_bit = board & (board - 1);
//         let removed_bit = board ^ board_without_first_bit;
//         let removed_bit_idx = single_bit_index(removed_bit);
//         result[removed_bit_idx] = 1.0;
//         board = board_without_first_bit;
//     }

//     result
// }

// fn single_bit_index(mut bit: u128) -> usize {
//     let mut n = 0;

//     if bit >> 64 != 0 {
//         n += 64;
//         bit >>= 64;
//     }
//     if bit >> 32 != 0 {
//         n += 32;
//         bit >>= 32;
//     }
//     if bit >> 16 != 0 {
//         n += 16;
//         bit >>= 16;
//     }
//     if bit >> 8 != 0 {
//         n += 8;
//         bit >>= 8;
//     }
//     if bit >> 4 != 0 {
//         n += 4;
//         bit >>= 4;
//     }
//     if bit >> 2 != 0 {
//         n += 2;
//         bit >>= 2;
//     }
//     if bit >> 1 != 0 {
//         n += 1;
//     }

//     return n;
// }
