use std::task::Poll;
use std::task::Context;
use std::pin::Pin;
use std::future::Future;
use pyo3::prelude::*;
use pyo3::types::{PyDict};

use super::super::analytics::{ActionWithPolicy,GameAnalytics,GameStateAnalysis};
use super::super::bits::single_bit_index;
use super::super::model::{self,TrainOptions};
use super::super::node_metrics::NodeMetrics;
use super::super::self_play::SelfPlaySample;
use super::engine::{GameState};
use super::action::{Action};

pub struct Model {
    name: String
}

impl Model {
    pub fn new(name: String) -> Self {
        Self {
            name
        }
    }
}

impl model::Model for Model {
    type State = GameState;
    type Action = Action;

    fn get_name(&self) -> &str {
        &self.name
    }

    #[allow(non_snake_case)]
    fn train(&self, target_name: &str, sample_metrics: &Vec<SelfPlaySample<Self::State, Self::Action>>, options: &TrainOptions) -> Model
    {
        let gil = Python::acquire_gil();
        let py = gil.python();

        let c4_model_module_name = "c4_model";
        let c4 = py.import(c4_model_module_name).unwrap();

        let X: Vec<_> = sample_metrics.iter().map(|v| game_state_to_input(&v.game_state)).collect();
        let yv: Vec<_> = sample_metrics.iter().map(|v| v.score).collect();
        let yp: Vec<_> = sample_metrics.iter().map(|v| map_policy_to_vec_input(&v.policy).to_vec()).collect();

        let py_options = PyDict::new(py);
        py_options.set_item("X", X).unwrap();
        py_options.set_item("yv", yv).unwrap();
        py_options.set_item("yp", yp).unwrap();

        py_options.set_item("train_ratio", options.train_ratio).unwrap();
        py_options.set_item("train_batch_size", options.train_batch_size).unwrap();
        py_options.set_item("epochs", options.epochs).unwrap();
        py_options.set_item("learning_rate", options.learning_rate).unwrap();
        py_options.set_item("policy_loss_weight", options.policy_loss_weight).unwrap();
        py_options.set_item("value_loss_weight", options.value_loss_weight).unwrap();

        c4.call("train", (&self.name, target_name), Some(py_options)).unwrap();

        Model::new(target_name.to_owned())
    }
}

impl GameAnalytics for Model {
    type Future = GameStateAnalysisFuture;
    type Action = Action;
    type State = GameState;

    /// Outputs a value from [-1, 1] depending on the player to move's evaluation of the current state.
    /// If the evaluation is a draw then 0.0 will be returned.
    /// Along with the value output a list of policy scores for all VALID moves is returned. If the position
    /// is terminal then the vector will be empty.
    fn get_state_analysis(&self, game_state: &GameState) -> GameStateAnalysisFuture {
        if let Some(value) = game_state.is_terminal() {
            return GameStateAnalysisFuture::new(GameStateAnalysis::new(
                Vec::new(),
                value
            ))
        }

        // @TODO: Add the cache back

        let input = game_state_to_input(game_state);
        let prediction = predict(&self.name, input).unwrap();
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

        GameStateAnalysisFuture::new(GameStateAnalysis::new(
            valid_actions_with_policies,
            prediction.0
        ))
    }
}

pub struct GameStateAnalysisFuture {
    output: Option<GameStateAnalysis<Action>>
}

impl GameStateAnalysisFuture {
    fn new(output: GameStateAnalysis<Action>) -> Self {
        Self { output: Some(output) }
    }
}

impl Future for GameStateAnalysisFuture {
    type Output = GameStateAnalysis<Action>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut s = self;
        Poll::Ready(s.output.take().unwrap())
    }
}

fn game_state_to_input(game_state: &GameState) -> Vec<Vec<Vec<f64>>> {
    let result: Vec<Vec<Vec<f64>>> = Vec::with_capacity(6);

    map_board_to_vec(game_state.p1_piece_board).iter()
        .zip(map_board_to_vec(game_state.p2_piece_board).iter())
        .enumerate()
        .fold(result, |mut r, (i, (p1, p2))| {
            let column_idx = i % 7;
            
            if column_idx == 0 {
                r.push(Vec::with_capacity(7))
            }

            let column_vec = r.last_mut().unwrap();

            column_vec.push(vec!(*p1, *p2));

            r
        })
}

fn predict(model_name: &str, model_input: Vec<Vec<Vec<f64>>>) -> PyResult<(f64, Vec<f64>)> {
    let gil = Python::acquire_gil();
    let py = gil.python();

    let c4_model_module_name = "c4_model";
    let c4 = py.import(c4_model_module_name)?;

    let result: (f64, Vec<f64>) = c4.call(
        "predict",
        (model_name, model_input),
        None
    )?.extract()?;

    Ok(result)
}

fn map_policy_to_vec_input(policy_metrics: &NodeMetrics<Action>) -> [f64; 7] {
    let total_visits = policy_metrics.visits as f64 - 1.0;
    let result:[f64; 7] = policy_metrics.children_visits.iter().fold([0.0; 7], |mut r, p| {
        match p.0 { Action::DropPiece(column) => r[column as usize - 1] = p.1 as f64 / total_visits };
        r
    });

    result
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