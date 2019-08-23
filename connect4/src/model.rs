use std::collections::HashMap;
use std::fs::{self,File};
use std::future::Future;
use std::pin::Pin;
use std::process::{Command,Stdio};
use std::sync::{Arc,Mutex,atomic::{AtomicBool,AtomicUsize,Ordering}};
use std::task::{Context,Poll,Waker};
use std::time::Instant;
use chashmap::{CHashMap};
use chrono::{Utc};
use crossbeam_queue::{SegQueue};
use failure::Error;
use reqwest::Client;
use serde::{Serialize,Deserialize};
use serde_json::json;

use model::analytics::{self,ActionWithPolicy,GameStateAnalysis};
use model::model::TrainOptions;
use model::model_info::ModelInfo;
use model::node_metrics::NodeMetrics;
use model::position_metrics::PositionMetrics;

use super::constants::{ANALYSIS_REQUEST_BATCH_SIZE,ANALYSIS_REQUEST_THREADS,DEPTH_TO_CACHE};
use super::paths::Paths;
use super::engine::{GameState};
use super::action::{Action};
use super::board::map_board_to_arr;

pub struct Model {
    model_info: ModelInfo,
    batching_model: Arc<BatchingModel>,
    alive: Arc<AtomicBool>,
    id_generator: Arc<AtomicUsize>
}

impl Model {
    pub fn new(model_info: ModelInfo) -> Self {
        let batching_model = Arc::new(BatchingModel::new(model_info.clone()));
        let alive = Arc::new(AtomicBool::new(true));

        for i in 0..ANALYSIS_REQUEST_THREADS {
            let batching_model_ref = batching_model.clone();
            let alive_ref = alive.clone();
            std::thread::spawn(move || {
                let mut last_report = Instant::now();
                loop {
                    let num_analysed = batching_model_ref.run_batch_predict();

                    if num_analysed == 0 {
                        std::thread::sleep(std::time::Duration::from_millis(1));
                    }

                    let elapsed_mills = last_report.elapsed().as_millis();
                    if i == 0 && elapsed_mills >= 5_000 {
                        let (num_nodes_from_cache, num_nodes_cache_miss, num_nodes) = batching_model_ref.take_num_nodes_analysed();
                        let (min_batch_size, max_batch_size) = batching_model_ref.take_min_max_batch_size();
                        let nps = num_nodes as f64 * 1000.0 / elapsed_mills as f64;
                        let cache_hit_perc = num_nodes_from_cache as f64 / (num_nodes_from_cache as f64 + num_nodes_cache_miss as f64) * 100.0;
                        let cache_coverage_perc = (num_nodes_from_cache as f64 + num_nodes_cache_miss as f64) / num_nodes as f64 * 100.0;
                        let state_analysis_cache_len = batching_model_ref.state_analysis_cache_len();
                        let now = Utc::now().format("%H:%M:%S").to_string();
                        println!(
                            "TIME: {}, NPS: {:.2}, Min Batch Size: {}, Max Batch Size: {}, Cache Size: {}, Cache Coverage: {:.2}%, Cache Hits: {:.2}%",
                            now,
                            nps,
                            min_batch_size,
                            max_batch_size,
                            state_analysis_cache_len,
                            cache_coverage_perc,
                            cache_hit_perc
                        );
                        last_report = Instant::now();
                    }

                    if !alive_ref.load(Ordering::SeqCst) {
                        break;
                    }

                    std::thread::sleep(std::time::Duration::from_micros(1));
                }
            });
        }

        Self {
            model_info,
            batching_model,
            alive,
            id_generator: Arc::new(AtomicUsize::new(0))
        }
    }
}

impl model::model::Model for Model {
    type State = GameState;
    type Action = Action;
    type Analyzer = GameAnalyzer;

    fn get_model_info(&self) -> &ModelInfo {
        &self.model_info
    }

    fn train(&self, target_model_info: ModelInfo, sample_metrics: &Vec<PositionMetrics<Self::State, Self::Action>>, options: &TrainOptions) -> Model
    {
        let model = train(&self.model_info, target_model_info, sample_metrics, options);

        model.expect("Failed to train model")
    }

    fn get_game_state_analyzer(&self) -> Self::Analyzer
    {
        GameAnalyzer {
            batching_model: self.batching_model.clone(),
            id_generator: self.id_generator.clone()
        }
    }
}

pub struct GameAnalyzer {
    batching_model: Arc<BatchingModel>,
    id_generator: Arc<AtomicUsize>
}

impl analytics::GameAnalyzer for GameAnalyzer {
    type State = GameState;
    type Action = Action;
    type Future = GameStateAnalysisFuture;

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

#[allow(non_snake_case)]
fn train(source_model_info: &ModelInfo, target_model_info: ModelInfo, sample_metrics: &Vec<PositionMetrics<GameState,Action>>, options: &TrainOptions) -> Result<Model, Error>
{
    println!("Training from {} to {}", source_model_info.get_model_name(), target_model_info.get_model_name());

    let X: Vec<_> = sample_metrics.iter().map(|v| game_state_to_input(&v.game_state)).collect();
    let yv: Vec<_> = sample_metrics.iter().map(|v| v.score).collect();
    let yp: Vec<_> = sample_metrics.iter().map(|v| map_policy_to_vec_input(&v.policy).to_vec()).collect();

    let json = json!({
        "x": X,
        "yv": yv,
        "yp": yp
    });

    let source_paths = Paths::from_model_info(&source_model_info);
    let source_base_path = source_paths.get_base_path();
    let train_data_path = source_base_path.join("training_data.json");

    serde_json::to_writer(
        &File::create(train_data_path.to_owned())?,
        &json
    )?;

    let docker_cmd = format!("docker run --rm \
        --runtime=nvidia \
        --mount type=bind,source=$(pwd)/{game_name}_runs,target=/{game_name}_runs \
        -e SOURCE_MODEL_PATH=/{game_name}_runs/{run_name}/models/{game_name}_{run_name}_{source_run_num:0>5}.h5 \
        -e TARGET_MODEL_PATH=/{game_name}_runs/{run_name}/models/{game_name}_{run_name}_{target_run_num:0>5}.h5 \
        -e EXPORT_MODEL_PATH=/{game_name}_runs/{run_name}/exported_models/{target_run_num} \
        -e TENSOR_BOARD_PATH=/{game_name}_runs/{run_name}/tensorboard \
        -e INITIAL_EPOCH={initial_epoch} \
        -e DATA_PATH=/{game_name}_runs/{run_name}/training_data.json \
        -e TRAIN_RATIO={train_ratio} \
        -e TRAIN_BATCH_SIZE={train_batch_size} \
        -e EPOCHS={epochs} \
        -e LEARNING_RATE={learning_rate} \
        -e POLICY_LOSS_WEIGHT={policy_loss_weight} \
        -e VALUE_LOSS_WEIGHT={value_loss_weight} \
        -e NVIDIA_VISIBLE_DEVICES=1 \
        quoridor_engine/train:latest",
        game_name = source_model_info.get_game_name(),
        run_name = source_model_info.get_run_name(),
        source_run_num = source_model_info.get_run_num(),
        target_run_num = target_model_info.get_run_num(),
        train_ratio = options.train_ratio,
        train_batch_size = options.train_batch_size,
        epochs = (source_model_info.get_run_num() - 1) + options.epochs,
        initial_epoch = (source_model_info.get_run_num() - 1),
        learning_rate = options.learning_rate,
        policy_loss_weight = options.policy_loss_weight,
        value_loss_weight = options.value_loss_weight
    );

    println!("{}", docker_cmd);

    let mut cmd = Command::new("/bin/bash")
        .arg("-c")
        .arg(docker_cmd)
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()?;

    let result = cmd.wait();

    println!("OUTPUT: {:?}", result);

    fs::remove_file(train_data_path)?;

    println!("Training process complete");

    Ok(Model::new(target_model_info))
}

impl Drop for Model {
    fn drop(&mut self) {
        self.alive.store(false, Ordering::SeqCst);
    }
}

pub struct BatchingModel {
    model_info: ModelInfo,
    states_to_analyse: SegQueue<(usize, GameState, Waker)>,
    states_analysed: Arc<Mutex<HashMap<usize, GameStateAnalysis<Action>>>>,
    state_analysis_cache: CHashMap<GameState, GameStateAnalysis<Action>>,
    num_nodes_analysed: AtomicUsize,
    num_nodes_from_cache: AtomicUsize,
    num_nodes_cache_miss: AtomicUsize,
    min_batch_size: AtomicUsize,
    max_batch_size: AtomicUsize
}

impl BatchingModel {
    fn new(model_info: ModelInfo) -> Self {
        let states_to_analyse = SegQueue::new();
        let states_analysed = Arc::new(Mutex::new(HashMap::with_capacity(ANALYSIS_REQUEST_BATCH_SIZE * ANALYSIS_REQUEST_THREADS)));
        let state_analysis_cache = CHashMap::with_capacity(7 * DEPTH_TO_CACHE + 1);
        let num_nodes_analysed = AtomicUsize::new(0);
        let num_nodes_from_cache = AtomicUsize::new(0);
        let num_nodes_cache_miss = AtomicUsize::new(0);
        let min_batch_size = AtomicUsize::new(std::usize::MAX);
        let max_batch_size = AtomicUsize::new(0);

        Self {
            model_info,
            states_to_analyse,
            states_analysed,
            state_analysis_cache,
            num_nodes_analysed,
            num_nodes_from_cache,
            num_nodes_cache_miss,
            min_batch_size,
            max_batch_size
        }
    }

    fn run_batch_predict(&self) -> usize {
        let states_to_analyse_queue = &self.states_to_analyse;

        let mut states_to_analyse: Vec<_> = Vec::with_capacity(ANALYSIS_REQUEST_BATCH_SIZE);
        while let Ok(state_to_analyse) = states_to_analyse_queue.pop() {
            states_to_analyse.push(state_to_analyse);

            if states_to_analyse.len() >= ANALYSIS_REQUEST_BATCH_SIZE {
                break;
            }
        }

        let states_to_analyse_len = states_to_analyse.len();
        self.min_batch_size.fetch_min(states_to_analyse_len, Ordering::SeqCst);
        self.max_batch_size.fetch_max(states_to_analyse_len, Ordering::SeqCst);

        if states_to_analyse_len == 0 {
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

    fn take_num_nodes_analysed(&self) -> (usize, usize, usize) {
        (    
           self.num_nodes_from_cache.swap(0, Ordering::SeqCst),
           self.num_nodes_cache_miss.swap(0, Ordering::SeqCst),
           self.num_nodes_analysed.swap(0, Ordering::SeqCst)
        )
    }

    fn take_min_max_batch_size(&self) -> (usize, usize) {
        (
            self.min_batch_size.swap(std::usize::MAX, Ordering::SeqCst),
            self.max_batch_size.swap(0, Ordering::SeqCst)
        )
    }

    fn state_analysis_cache_len(&self) -> usize {
        self.state_analysis_cache.len()
    }

    fn poll(&self, id: usize, game_state: &GameState, waker: &Waker) -> Poll<GameStateAnalysis<Action>> {
        if let Some(value) = game_state.is_terminal() {
            self.num_nodes_analysed.fetch_add(1, Ordering::SeqCst);
            return Poll::Ready(GameStateAnalysis::new(
                value,
                Vec::new()
            ));
        }
        
        let analysis = self.states_analysed.lock().unwrap().remove(&id);

        match analysis {
            Some(analysis) => {
                self.num_nodes_analysed.fetch_add(1, Ordering::SeqCst);
                Poll::Ready(analysis)
            },
            None => {
                if game_state.number_of_actions() <= DEPTH_TO_CACHE {
                    if let Some(analysis) = self.state_analysis_cache.get(game_state) {
                        self.num_nodes_analysed.fetch_add(1, Ordering::SeqCst);
                        self.num_nodes_from_cache.fetch_add(1, Ordering::SeqCst);
                        return Poll::Ready(analysis.to_owned());
                    } else {
                        self.num_nodes_cache_miss.fetch_add(1, Ordering::SeqCst);
                    }
                }

                self.states_to_analyse.push((
                    id,
                    game_state.to_owned(),
                    waker.clone()
                ));
                Poll::Pending
            }
        }
    }

    fn predict(&self, game_states: Vec<&GameState>) -> Result<Vec<GameStateAnalysis<Action>>, Error> {
        let body = game_states_to_request_body(&game_states);

        let request_url = get_model_url(&self.model_info);

        let mut response;

        loop {
            response = Client::new()
                .post(&request_url)
                .json(&body)
                .send();

            match &response {
                Err(_) => (),
                Ok(response) if response.status().is_success() => break,
                _ => ()
            }

            println!("Failed to make a http request to prediction: {}", &request_url);

            std::thread::sleep(std::time::Duration::from_secs(10));
        }

        let predictions: PredictionResults = response?.json()?;

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

impl Drop for BatchingModel {
    fn drop(&mut self) {
        println!("Dropping BatchingModel: {}", self.model_info.get_model_name());
        println!("states_to_analyse length: {}", self.states_to_analyse.len());
        println!("states_analysed length: {}", self.states_analysed.lock().unwrap().len());
        println!("state_analysis_cache length: {}", self.state_analysis_cache.len());
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

fn get_model_url(model_info: &ModelInfo) -> String {
    format!(
        "http://localhost:8501/v1/models/exported_models/versions/{version}:predict",
        version = model_info.get_run_num()
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

    map_board_to_arr(game_state.p1_piece_board).iter()
        .zip(map_board_to_arr(game_state.p2_piece_board).iter())
        .enumerate()
        .fold(result, |mut r, (i, (p1, p2))| {
            let column_idx = i % 7;
            
            if column_idx == 0 {
                r.push(Vec::with_capacity(7))
            }

            let column_vec = r.last_mut().unwrap();

            // The input is normalized by listing the player to move first. This is different than having
            // black first and then red. So on red's turn, red will be listed first, then black.
            let (c1, c2) = if game_state.p1_turn_to_move {
                (*p1, *p2)
            } else {
                (*p2, *p1)
            };

            column_vec.push(vec!(c1, c2));

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
