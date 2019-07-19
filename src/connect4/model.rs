use std::collections::HashMap;
use std::sync::{Arc,Mutex,atomic::{AtomicBool,AtomicUsize,Ordering}};
use std::task::{Context,Poll,Waker};
use std::future::Future;
use std::pin::Pin;
use std::time::Instant;
use chashmap::{CHashMap};
use chrono::{Utc};
use reqwest::Client;
use serde::{Serialize,Deserialize};
use serde_json::json;
use pyo3::prelude::*;
use pyo3::types::{PyDict};
use crossbeam_queue::{SegQueue};

use super::super::analytics::{ActionWithPolicy,GameAnalytics,GameStateAnalysis};
use super::super::bits::single_bit_index;
use super::super::model::{self,TrainOptions};
use super::super::node_metrics::NodeMetrics;
use super::super::self_play::SelfPlaySample;
use super::engine::{GameState};
use super::action::{Action};

const DEPTH_TO_CACHE: usize = 7;

pub struct Model {
    name: String,
    batching_model: Arc<BatchingModel>,
    alive: Arc<AtomicBool>,
    id_generator: AtomicUsize
}

impl Model {
    pub fn new(name: String) -> Self {
        let batching_model = Arc::new(BatchingModel::new(name.to_owned()));
        let alive = Arc::new(AtomicBool::new(true));

        for i in 0..3 {
            let batching_model_ref = batching_model.clone();
            let alive_ref = alive.clone();
            std::thread::spawn(move || {
                let mut last_report = Instant::now();
                loop {
                    let num_analysed = batching_model_ref.run_predict();

                    if num_analysed == 0 {
                        std::thread::sleep(std::time::Duration::from_millis(1));
                    }

                    let elapsed_mills = last_report.elapsed().as_millis();
                    if i == 0 && elapsed_mills >= 5_000 {
                        let num_nodes = batching_model_ref.take_num_nodes_analysed();
                        let nps = num_nodes as f64 * 1000.0 / elapsed_mills as f64;
                        let now = Utc::now().format("%H:%M:%S").to_string();
                        println!("TIME: {}, NPS: {:.2}", now, nps);
                        last_report = Instant::now();
                    }

                    if !alive_ref.load(Ordering::SeqCst) {
                        break;
                    }
                }
            });
        }

        Self {
            name,
            batching_model,
            alive,
            id_generator: AtomicUsize::new(0)
        }
    }
}

impl model::Model for Model {
    type State = GameState;
    type Action = Action;

    fn get_name(&self) -> &str {
        &self.name
    }

    fn train(&self, target_name: &str, sample_metrics: &Vec<SelfPlaySample<Self::State, Self::Action>>, options: &TrainOptions) -> Model
    {
        let model = train(&self.name, target_name, sample_metrics, options);

        model.map_err(|e| {
            let gil = Python::acquire_gil();
            let py = gil.python();
            e.print(py);
        }).expect("Failed to train model")
    }
}

#[allow(non_snake_case)]
fn train(source_name: &str, target_name: &str, sample_metrics: &Vec<SelfPlaySample<GameState,Action>>, options: &TrainOptions) -> PyResult<Model>
{
    let gil = Python::acquire_gil();
    let py = gil.python();

    let c4_model_module_name = "c4_model";
    let c4 = py.import(c4_model_module_name)?;

    let X: Vec<_> = sample_metrics.iter().map(|v| game_state_to_input(&v.game_state)).collect();
    let yv: Vec<_> = sample_metrics.iter().map(|v| v.score).collect();
    let yp: Vec<_> = sample_metrics.iter().map(|v| map_policy_to_vec_input(&v.policy).to_vec()).collect();

    let py_options = PyDict::new(py);
    py_options.set_item("X", X)?;
    py_options.set_item("yv", yv)?;
    py_options.set_item("yp", yp)?;

    py_options.set_item("train_ratio", options.train_ratio)?;
    py_options.set_item("train_batch_size", options.train_batch_size)?;
    py_options.set_item("epochs", options.epochs)?;
    py_options.set_item("learning_rate", options.learning_rate)?;
    py_options.set_item("policy_loss_weight", options.policy_loss_weight)?;
    py_options.set_item("value_loss_weight", options.value_loss_weight)?;

    c4.call("train", (source_name, target_name), Some(py_options))?;

    Ok(Model::new(target_name.to_owned()))
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
            self.id_generator.fetch_add(1, Ordering::SeqCst),
            self.batching_model.clone()
        )
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        self.alive.store(false, Ordering::SeqCst);
    }
}

pub struct BatchingModel {
    model_name: String,
    states_to_analyse: SegQueue<(usize, GameState, Waker)>,
    states_analysed: Arc<Mutex<HashMap<usize, GameStateAnalysis<Action>>>>,
    state_analysis_cache: CHashMap<GameState, GameStateAnalysis<Action>>,
    num_nodes_analysed: AtomicUsize
}

impl BatchingModel {
    fn new(model_name: String) -> Self {
        let states_to_analyse = SegQueue::new();
        let states_analysed = Arc::new(Mutex::new(HashMap::with_capacity(4_092)));
        let state_analysis_cache = CHashMap::with_capacity(7 * DEPTH_TO_CACHE);
        let num_nodes_analysed = AtomicUsize::new(0);

        Self {
            model_name,
            states_to_analyse,
            states_analysed,
            state_analysis_cache,
            num_nodes_analysed
        }
    }

    fn run_predict(&self) -> usize {
        let states_to_analyse_queue = &self.states_to_analyse;

        let mut states_to_analyse: Vec<_> = Vec::with_capacity(1024);
        while let Ok(state_to_analyse) = states_to_analyse_queue.pop() {
            states_to_analyse.push(state_to_analyse);

            if states_to_analyse.len() >= 1024 {
                break;
            }
        }

        if states_to_analyse.len() == 0 {
            return 0;
        }

        let game_states_to_predict = states_to_analyse.iter().map(|(_,s,_)| s).collect();
        let analysis: Vec<_> = self.predict(game_states_to_predict).unwrap();
        let num_analysed = analysis.len();

        for ((id, s, waker), analysis) in states_to_analyse.into_iter().zip(analysis) {
            if s.number_of_actions() <= DEPTH_TO_CACHE {
                self.state_analysis_cache.insert(s, analysis.to_owned());
            }

            self.states_analysed.lock().unwrap().insert(id, analysis);
            waker.wake();
        }

        num_analysed
    }

    fn take_num_nodes_analysed(&self) -> usize {
        self.num_nodes_analysed.swap(0, Ordering::SeqCst)
    }

    fn poll(&self, id: usize, game_state: &GameState, waker: &Waker) -> Poll<GameStateAnalysis<Action>> {
        if let Some(value) = game_state.is_terminal() {
            self.num_nodes_analysed.fetch_add(1, Ordering::SeqCst);
            return Poll::Ready(GameStateAnalysis::new(
                value,
                Vec::new()
            ));
        }

        if game_state.number_of_actions() <= DEPTH_TO_CACHE {
            if let Some(analysis) = self.state_analysis_cache.get(game_state) {
                self.num_nodes_analysed.fetch_add(1, Ordering::SeqCst);
                return Poll::Ready(analysis.to_owned());
            }
        }
        
        let analysis = self.states_analysed.lock().unwrap().remove(&id);

        match analysis {
            Some(analysis) => {
                self.num_nodes_analysed.fetch_add(1, Ordering::SeqCst);
                Poll::Ready(analysis)
            },
            None => {
                self.states_to_analyse.push((
                    id,
                    game_state.to_owned(),
                    waker.clone()
                ));
                Poll::Pending
            }
        }
    }

    fn predict(&self, game_states: Vec<&GameState>) -> Result<Vec<GameStateAnalysis<Action>>, &'static str> {
        let body = game_states_to_request_body(&game_states);

        let request_url = get_model_url(&self.model_name);
        let mut response = Client::new()
            .post(&request_url)
            .json(&body)
            .send()
            .map_err(|_| "Failed to make a http request to prediction")?;

        let predictions: PredictionResults = response.json()
            .map_err(|_| "Failed to deserialize predict http result")?;

        let result = game_states.iter()
            .zip(predictions.predictions.into_iter())
            .map(|(game_state, result)| {
                let value_score = result.get("value_head/Tanh:0").unwrap()[0];
                let policy_scores = result.get("policy_head/Softmax:0").unwrap();

                let valid_actions_with_policies: Vec<ActionWithPolicy<Action>> = game_state.get_valid_actions().iter()
                    .zip(policy_scores).enumerate()
                    .filter_map(|(i, (v, p))|
                    {
                        if *v {
                            Some(ActionWithPolicy::new(
                                Action::DropPiece((i + 1) as u64),
                                *p
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
            .collect();

        Ok(result)
    }
}

pub struct GameStateAnalysisFuture {
    game_state: GameState,
    id: usize,
    batching_model: Arc<BatchingModel>
}

impl GameStateAnalysisFuture {
    fn new(
        game_state: GameState,
        id: usize,
        batching_model: Arc<BatchingModel>
    ) -> Self
    {
        Self { game_state, id, batching_model }
    }
}

impl Future for GameStateAnalysisFuture {
    type Output = GameStateAnalysis<Action>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        (*self).batching_model.poll(self.id, &self.game_state, cx.waker())
    }
}

#[derive(Serialize)]
struct RequestImage {
    input_image: Vec<Vec<Vec<f64>>>
}

#[derive(Debug, Deserialize)]
struct PredictionResults {
    predictions: Vec<HashMap<String,Vec<f64>>>
}

fn get_model_url(model_name: &str) -> String {
    let split_name: Vec<_> = model_name.split('_').collect();
    format!(
        "http://localhost:8501/v1/models/{run_name}/versions/{version}:predict",
        run_name = split_name[1],
        version = split_name[2].parse::<usize>().unwrap()
    )
}

fn game_states_to_request_body(game_states: &Vec<&GameState>) -> serde_json::value::Value {
    let game_states: Vec<_> = game_states.iter().map(|game_state| RequestImage {
        input_image: game_state_to_input(game_state)
    }).collect();

    json!({
        "instances": game_states
    })
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