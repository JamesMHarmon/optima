use std::sync::Arc;
use std::task::{Context,Poll,Waker};
use std::future::Future;
use std::pin::Pin;
use chashmap::{CHashMap};
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
    name: String,
    batching_model: Arc<BatchingModel>
}

impl Model {
    pub fn new(name: String) -> Self {
        let batching_model = Arc::new(BatchingModel::new(name.to_owned()));
        let batching_model_ref = batching_model.clone();

        // @TODO: Add logic to destroy thread.
        println!("Creating Thread");
        std::thread::spawn(move || {
            loop {
                let num_analysed = batching_model_ref.run_predict();

                if num_analysed == 0 {
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
            }
        });

        Self {
            name,
            batching_model
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
        GameStateAnalysisFuture::new(
            game_state.to_owned(),
            self.batching_model.clone()
        )
    }
}

pub struct BatchingModel {
    model_name: String,
    states_to_analyse: CHashMap<GameState, Vec<Waker>>,
    state_analysis_cache: CHashMap<GameState, GameStateAnalysis<Action>>
}

impl BatchingModel {
    fn new(model_name: String) -> Self {
        let states_to_analyse = CHashMap::with_capacity(512);
        let state_analysis_cache = CHashMap::with_capacity(2_000_000);

        Self {
            model_name,
            states_to_analyse,
            state_analysis_cache
        }
    }

    fn run_predict(&self) -> usize {
        let entries: Vec<_> = self.states_to_analyse.clone().into_iter().collect();

        if entries.len() == 0 {
            return 0;
        }

        self.states_to_analyse.clear();

        let analysis: Vec<_> = self.predict(entries.iter().map(|(s,_)| s).collect());
        let num_analysed = analysis.len();

        println!("Analysed: {}", num_analysed);

        for ((s, wakers), analysis) in entries.into_iter().zip(analysis) {
            self.state_analysis_cache.insert(s.to_owned(), analysis);
            for w in wakers {
                w.wake();
            }
        }

        num_analysed
    }

    fn poll(&self, game_state: &GameState, waker: &Waker) -> Poll<GameStateAnalysis<Action>> {
        if let Some(value) = game_state.is_terminal() {
            return Poll::Ready(GameStateAnalysis::new(
                value,
                Vec::new()
            ));
        }
        
        let analysis = self.state_analysis_cache.get(game_state);

        match analysis {
            Some(analysis) => Poll::Ready(analysis.to_owned()),
            None => {
                self.register(game_state, waker.clone());
                Poll::Pending
            }
        }
    }

    fn register(&self, game_state: &GameState, waker: Waker) {
        let w1 = waker.clone();
        self.states_to_analyse.upsert(
            game_state.to_owned(),
            || vec!(w1),
            |v| v.push(waker)
        );
    }

    fn predict(&self, game_states: Vec<&GameState>) -> Vec<GameStateAnalysis<Action>> {
        let input = game_states_to_input(&game_states);
        let predictions = predict(&self.model_name, input).unwrap();

        game_states.iter()
            .zip(predictions.into_iter())
            .map(|(game_state, (value_score, policy_scores))| {
                let valid_actions_with_policies: Vec<ActionWithPolicy<Action>> = game_state.get_valid_actions().iter()
                    .zip(policy_scores).enumerate()
                    .filter_map(|(i, (v, p))|
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

                GameStateAnalysis {
                    policy_scores: valid_actions_with_policies,
                    value_score
                }
            })
            .collect()
    }
}

pub struct GameStateAnalysisFuture {
    game_state: GameState,
    batching_model: Arc<BatchingModel>
}

impl GameStateAnalysisFuture {
    fn new(
        game_state: GameState,
        batching_model: Arc<BatchingModel>
    ) -> Self
    {
        Self { game_state, batching_model }
    }
}

impl Future for GameStateAnalysisFuture {
    type Output = GameStateAnalysis<Action>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        (*self).batching_model.poll(&self.game_state, cx.waker())
    }
}

fn game_states_to_input(game_states: &Vec<&GameState>) -> Vec<Vec<Vec<Vec<f64>>>> {
    game_states.iter().map(|game_state| game_state_to_input(game_state)).collect()
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

fn predict(model_name: &str, model_input: Vec<Vec<Vec<Vec<f64>>>>) -> PyResult<Vec<(f64, Vec<f64>)>> {
    let gil = Python::acquire_gil();
    let py = gil.python();

    let c4_model_module_name = "c4_model";
    let c4 = py.import(c4_model_module_name)?;

    let result: Vec<(f64, Vec<f64>)> = c4.call(
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