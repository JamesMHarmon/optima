use pyo3::prelude::*;

use super::super::analytics::{ActionWithPolicy,GameAnalytics,GameStateAnalysis};
use super::super::bits::single_bit_index;
use super::super::model::{self, TrainOptions};
use super::engine::{GameState};
use super::action::{Action};

pub struct Model {
    model_name: String
}

impl Model {
    pub fn new() -> Self {
        Self {
            model_name: "FIX_THIS".to_string()
        }
    }
}


impl model::Model for Model {
    fn create(&mut self, name: &str) {}
    fn train(&mut self, from_name: &str, target_name: &str, options: &TrainOptions) {}
}

impl GameAnalytics<GameState, Action> for Model {
    /// Outputs a value from [-1, 1] depending on the player to move's evaluation of the current state.
    /// If the evaluation is a draw then 0.0 will be returned.
    /// Along with the value output a list of policy scores for all VALID moves is returned. If the position
    /// is terminal then the vector will be empty.
    fn get_state_analysis(&self, game_state: &GameState) -> GameStateAnalysis<Action> {
        if let Some(value) = game_state.is_terminal() {
            return GameStateAnalysis::new(
                Vec::new(),
                value
            )
        }

        let input = game_state_to_input(game_state);
        let prediction = predict(&self.model_name, &input).unwrap();
        let valid_actions_with_policies: Vec<ActionWithPolicy<Action>> = game_state.get_valid_actions().iter().zip(prediction.1).enumerate().filter_map(|(i, (v, p))|
        {
            if *v {
                Some(ActionWithPolicy::new(
                    Action::DropPiece((i + 1) as u64),
                    p
                ))
            } else {
                None
            }
        }).collect();

        GameStateAnalysis::new(
            valid_actions_with_policies,
            prediction.0
        )
    }
}

fn game_state_to_input(game_state: &GameState) -> (Vec<f64>, Vec<f64>) {
    (
        map_board_to_vec(game_state.p1_piece_board).to_vec(),
        map_board_to_vec(game_state.p2_piece_board).to_vec()
    )
}

fn predict(model_name: &str, model_input: &(Vec<f64>, Vec<f64>)) -> PyResult<(f64, Vec<f64>)> {
    let gil = Python::acquire_gil();
    let py = gil.python();

    let c4_model_module_name = "c4_model";
    let c4 = py.import(c4_model_module_name)?;

    let result: (f64, Vec<f64>) = c4.call(
        "analyse",
        (model_name, model_input.0.to_owned(), model_input.1.to_owned()),
        None
    )?.extract()?;

    Ok(result)
}

fn map_board_to_vec(mut board: u64) -> [f64; 42] {
    let mut result:[f64; 42] = [0.0; 42];
    while board != 0 {
        let board_without_first_bit = board & (board - 1);
        let removed_bit = board ^ board_without_first_bit;
        let removed_bit_idx = single_bit_index(removed_bit as u128);
        let removed_bit_vec_idx = map_board_idx_to_vec_idx(removed_bit_idx);
    
        result[removed_bit_vec_idx] = 1.0;
        board = board_without_first_bit;
    }

    result
}

/// Converts from the bit_board index, which starts in the bottom left and traverses bottom to top with every
/// 7th bit being empty.
/// From:
/// 05  12  19  26  33  40  47
/// 04  11  18  25  32  39  46
/// 03  10  17  24  31  38  45
/// 02  09  16  23  30  37  44
/// 01  08  15  22  29  36  43
/// 00  07  14  21  28  35  42
///
/// To the vector form which start in the top left and goes left to right with no skipped bits.
/// To:
/// 00  01  02  03  04  05  06
/// 07  08  09  10  11  12  13
/// 14  15  16  17  18  19  20
/// 21  22  23  24  25  26  27
/// 28  29  30  31  32  33  34
/// 35  36  37  38  39  40  41
fn map_board_idx_to_vec_idx(board_idx: usize) -> usize {
    let removed_bit_pos = board_idx + 1;
    let column_idx = removed_bit_pos / 7;
    let row_idx = (removed_bit_pos % 7) - 1;
    ((5 - row_idx) * 7) + column_idx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_map_board_idx_to_vec_idx_bottom_left() {
        let board_idx = 0;
        let expected_vec_idx = 35;
        let actual_vec_index = map_board_idx_to_vec_idx(board_idx);
        assert_eq!(expected_vec_idx, actual_vec_index);
    }

    #[test]
    fn test_map_board_idx_to_vec_idx_top_left() {
        let board_idx = 5;
        let expected_vec_idx = 0;
        let actual_vec_index = map_board_idx_to_vec_idx(board_idx);
        assert_eq!(expected_vec_idx, actual_vec_index);
    }

    #[test]
    fn test_map_board_idx_to_vec_idx_bottom_right() {
        let board_idx = 42;
        let expected_vec_idx = 41;
        let actual_vec_index = map_board_idx_to_vec_idx(board_idx);
        assert_eq!(expected_vec_idx, actual_vec_index);
    }

    #[test]
    fn test_map_board_idx_to_vec_idx_top_right() {
        let board_idx = 47;
        let expected_vec_idx = 06;
        let actual_vec_index = map_board_idx_to_vec_idx(board_idx);
        assert_eq!(expected_vec_idx, actual_vec_index);
    }
}