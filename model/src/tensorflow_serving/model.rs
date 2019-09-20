use std::hash::Hash;
use std::collections::HashMap;
use std::fs::{self,File};
use std::future::Future;
use std::pin::Pin;
use std::process::{Command,Stdio};
use std::sync::{Arc,Mutex,atomic::{AtomicBool,AtomicUsize,Ordering}};
use std::task::{Context,Poll,Waker};
use std::time::Instant;
use chrono::{Utc};
use crossbeam_queue::{SegQueue};
use failure::Error;
use itertools::Itertools;
use reqwest::Client;
use serde::{Serialize,Deserialize};
use serde_json::json;

use engine::game_state::GameState;
use engine::engine::GameEngine;

use super::constants::{ANALYSIS_REQUEST_BATCH_SIZE,ANALYSIS_REQUEST_THREADS,TRAIN_DATA_CHUNK_SIZE};
use super::paths::Paths;
use super::super::analytics::{self,ActionWithPolicy,GameStateAnalysis};
use super::super::model::{Model as ModelTrait,TrainOptions};
use super::super::model_info::ModelInfo;
use super::super::node_metrics::NodeMetrics;
use super::super::position_metrics::PositionMetrics;

pub struct TensorflowServingModel<S,A,E,Map>
{
    model_info: ModelInfo,
    batching_model: Arc<BatchingModel<S,A,E,Map>>,
    alive: Arc<AtomicBool>,
    id_generator: Arc<AtomicUsize>,
    mapper: Arc<Map>
}

pub trait Mapper<S,A> {
    fn game_state_to_input(&self, game_state: &S) -> Vec<Vec<Vec<f32>>>;
    fn policy_metrics_to_expected_output(&self, policy: &NodeMetrics<A>) -> Vec<f32>;
    fn policy_to_valid_actions(&self, game_state: &S, policy_scores: &Vec<f32>) -> Vec<ActionWithPolicy<A>>;
}

impl<S,A,E,Map> TensorflowServingModel<S,A,E,Map>
where
    S: Clone + PartialEq + Hash + Send + Sync + 'static,
    A: Clone + Send + Sync + 'static,
    E: GameEngine<State=S,Action=A> + Send + Sync + 'static,
    Map: Mapper<S,A> + Send + Sync + 'static
{
    pub fn new(
        model_info: ModelInfo,
        engine: E,
        mapper: Map
    ) -> Self
    {
        let mapper = Arc::new(mapper);
        let batching_model = Arc::new(BatchingModel::new(model_info.clone(), engine, mapper.clone()));
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
                        let num_nodes = batching_model_ref.take_num_nodes_analysed();
                        let (min_batch_size, max_batch_size) = batching_model_ref.take_min_max_batch_size();
                        let nps = num_nodes as f32 * 1000.0 / elapsed_mills as f32;
                        let now = Utc::now().format("%H:%M:%S").to_string();
                        println!(
                            "TIME: {}, NPS: {:.2}, Min Batch Size: {}, Max Batch Size: {}",
                            now,
                            nps,
                            min_batch_size,
                            max_batch_size
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
            id_generator: Arc::new(AtomicUsize::new(0)),
            mapper
        }
    }

    pub fn create(
        model_info: &ModelInfo,
        num_filters: usize,
        num_blocks: usize,
        (input_h, input_w, input_c): (usize, usize, usize),
        output_size: usize
     ) -> Result<(), Error> {
        create(
            model_info,
            num_filters,
            num_blocks,
            (input_h, input_w, input_c),
            output_size
        )
    }
}

impl<S,A,E,Map> ModelTrait for TensorflowServingModel<S,A,E,Map>
where
    S: GameState + Send + Sync + 'static,
    A: Clone + Send + Sync + 'static,
    E: GameEngine<State=S,Action=A> + Send + Sync + 'static,
    Map: Mapper<S,A> + Send + Sync + 'static
{
    type State = S;
    type Action = A;
    type Analyzer = GameAnalyzer<S,A,E,Map>;

    fn get_model_info(&self) -> &ModelInfo {
        &self.model_info
    }

    fn train<I>(
        &self,
        target_model_info: &ModelInfo,
        sample_metrics: I,
        options: &TrainOptions
    ) -> Result<(), Error>
    where
        I: Iterator<Item=PositionMetrics<S,A>>
    {
        let mapper = &*self.mapper;

        train(&self.model_info, target_model_info, sample_metrics, mapper, options)
    }

    fn get_game_state_analyzer(&self) -> Self::Analyzer
    {
        GameAnalyzer {
            batching_model: self.batching_model.clone(),
            id_generator: self.id_generator.clone()
        }
    }
}

pub struct GameAnalyzer<S,A,E,Map> {
    batching_model: Arc<BatchingModel<S,A,E,Map>>,
    id_generator: Arc<AtomicUsize>
}

impl<S,A,E,Map> analytics::GameAnalyzer for GameAnalyzer<S,A,E,Map>
where
    S: Clone + PartialEq + Hash,
    A: Clone,
    E: GameEngine<State=S,Action=A> + Send + Sync + 'static,
    Map: Mapper<S,A>
{
    type State = S;
    type Action = A;
    type Future = GameStateAnalysisFuture<S,A,E,Map>;

    /// Outputs a value from [-1, 1] depending on the player to move's evaluation of the current state.
    /// If the evaluation is a draw then 0.0 will be returned.
    /// Along with the value output a list of policy scores for all VALID moves is returned. If the position
    /// is terminal then the vector will be empty.
    fn get_state_analysis(&self, game_state: &S) -> GameStateAnalysisFuture<S,A,E,Map> {
        GameStateAnalysisFuture::new(
            game_state.to_owned(),
            self.id_generator.fetch_add(1, Ordering::SeqCst),
            self.batching_model.clone()
        )
    }
}

#[allow(non_snake_case)]
fn train<S,A,I,Map>(
    source_model_info: &ModelInfo,
    target_model_info: &ModelInfo,
    sample_metrics: I,
    mapper: &Map,
    options: &TrainOptions
) -> Result<(), Error>
where
    S: GameState + Send + Sync + 'static,
    A: Clone + Send + Sync + 'static,
    I: Iterator<Item=PositionMetrics<S,A>>,
    Map: Mapper<S,A>
{
    println!("Training from {} to {}", source_model_info.get_model_name(), target_model_info.get_model_name());
    
    let mut train_data_file_names = vec!();

    for (i, sample_metrics) in sample_metrics.chunks(TRAIN_DATA_CHUNK_SIZE).into_iter().enumerate() {
        let sample_metrics: Vec<_> = sample_metrics.collect();
        let X: Vec<_> = sample_metrics.iter().map(|v| mapper.game_state_to_input(&v.game_state)).collect();
        let yv: Vec<_> = sample_metrics.iter().map(|v| v.score).collect();
        let yp: Vec<_> = sample_metrics.iter().map(|v| mapper.policy_metrics_to_expected_output(&v.policy)).collect();

        let json = json!({
            "x": X,
            "yv": yv,
            "yp": yp
        });

        let source_paths = Paths::from_model_info(&source_model_info);
        let source_base_path = source_paths.get_base_path();
        let train_data_file_name = format!("training_data_{}.json", i);
        let train_data_path = source_base_path.join(&train_data_file_name);

        println!("Writing data to {:?}", &train_data_path);

        serde_json::to_writer(
            &File::create(train_data_path.to_owned())?,
            &json
        )?;

        train_data_file_names.push(train_data_file_name);
    }

    let train_data_paths = train_data_file_names.iter().map(|file_name| format!(
        "/{game_name}_runs/{run_name}/{file_name}",
        game_name = source_model_info.get_game_name(),
        run_name = source_model_info.get_run_name(),
        file_name = file_name
    ));

    let docker_cmd = format!("docker run --rm \
        --runtime=nvidia \
        --mount type=bind,source=\"$(pwd)/{game_name}_runs\",target=/{game_name}_runs \
        -e SOURCE_MODEL_PATH=/{game_name}_runs/{run_name}/models/{game_name}_{run_name}_{source_run_num:0>5}.h5 \
        -e TARGET_MODEL_PATH=/{game_name}_runs/{run_name}/models/{game_name}_{run_name}_{target_run_num:0>5}.h5 \
        -e EXPORT_MODEL_PATH=/{game_name}_runs/{run_name}/exported_models/{target_run_num} \
        -e TENSOR_BOARD_PATH=/{game_name}_runs/{run_name}/tensorboard \
        -e INITIAL_EPOCH={initial_epoch} \
        -e DATA_PATHS={train_data_paths} \
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
        train_data_paths = train_data_paths.map(|p| format!("\"{}\"", p)).join(","),
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

    for path in train_data_file_names {
        fs::remove_file(path)?;
    }

    println!("Training process complete");

    Ok(())
}

#[allow(non_snake_case)]
fn create(
    model_info: &ModelInfo,
    num_filters: usize,
    num_blocks: usize,
    (input_h, input_w, input_c): (usize, usize, usize),
    output_size: usize
) -> Result<(), Error>
{
    let game_name = model_info.get_game_name();
    let run_name = model_info.get_run_name();

    fs::create_dir_all(format!(
        "./{game_name}_runs/{run_name}/models",
        game_name = game_name,
        run_name = run_name
    ))?;

    let docker_cmd = format!("docker run --rm \
        --runtime=nvidia \
        --mount type=bind,source=\"$(pwd)/{game_name}_runs\",target=/{game_name}_runs \
        -e TARGET_MODEL_PATH=/{game_name}_runs/{run_name}/models/{game_name}_{run_name}_00001.h5 \
        -e EXPORT_MODEL_PATH=/{game_name}_runs/{run_name}/exported_models/1 \
        -e INPUT_H={input_h} \
        -e INPUT_W={input_w} \
        -e INPUT_C={input_c} \
        -e OUTPUT_SIZE={output_size} \
        -e NUM_FILTERS={num_filters} \
        -e NUM_BLOCKS={num_blocks} \
        -e NVIDIA_VISIBLE_DEVICES=1 \
        quoridor_engine/create:latest",
        game_name = game_name,
        run_name = run_name,
        input_h = input_h,
        input_w = input_w,
        input_c = input_c,
        output_size = output_size,
        num_filters = num_filters,
        num_blocks = num_blocks,
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

    println!("Model creation process complete");

    Ok(())
}

impl<S,A,E,Map> Drop for TensorflowServingModel<S,A,E,Map> {
    fn drop(&mut self) {
        self.alive.store(false, Ordering::SeqCst);
    }
}

pub struct BatchingModel<S,A,E,Map> {
    model_info: ModelInfo,
    states_to_analyse: SegQueue<(usize, S, Waker)>,
    states_analysed: Arc<Mutex<HashMap<usize, GameStateAnalysis<A>>>>,
    num_nodes_analysed: AtomicUsize,
    min_batch_size: AtomicUsize,
    max_batch_size: AtomicUsize,
    engine: E,
    mapper: Arc<Map>
}

impl<S,A,E,Map> BatchingModel<S,A,E,Map>
where
    S: Clone + PartialEq + Hash,
    A: Clone,
    E: GameEngine<State=S,Action=A> + Send + Sync + 'static,
    Map: Mapper<S,A>
{
    fn new(model_info: ModelInfo, engine: E, mapper: Arc<Map>) -> Self
    {
        let states_to_analyse = SegQueue::new();
        let states_analysed = Arc::new(Mutex::new(HashMap::with_capacity(ANALYSIS_REQUEST_BATCH_SIZE * ANALYSIS_REQUEST_THREADS)));
        let num_nodes_analysed = AtomicUsize::new(0);
        let min_batch_size = AtomicUsize::new(std::usize::MAX);
        let max_batch_size = AtomicUsize::new(0);

        Self {
            model_info,
            states_to_analyse,
            states_analysed,
            num_nodes_analysed,
            min_batch_size,
            max_batch_size,
            engine,
            mapper
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

        for ((id, _s, waker), analysis) in states_to_analyse.into_iter().zip(analysis) {
            self.states_analysed.lock().unwrap().insert(id, analysis);
            waker.wake();
        }

        num_analysed
    }

    fn take_num_nodes_analysed(&self) -> usize {
        (
           self.num_nodes_analysed.swap(0, Ordering::SeqCst)
        )
    }

    fn take_min_max_batch_size(&self) -> (usize, usize) {
        (
            self.min_batch_size.swap(std::usize::MAX, Ordering::SeqCst),
            self.max_batch_size.swap(0, Ordering::SeqCst)
        )
    }

    fn poll(&self, id: usize, game_state: &S, waker: &Waker) -> Poll<GameStateAnalysis<A>> {
        let is_terminal = self.engine.is_terminal_state(game_state);

        if let Some(value) = is_terminal {
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
                self.states_to_analyse.push((
                    id,
                    game_state.clone(),
                    waker.clone()
                ));

                Poll::Pending
            }
        }
    }

    fn predict(&self, game_states: Vec<&S>) -> Result<Vec<GameStateAnalysis<A>>, Error> {
        let body = game_states_to_request_body(&game_states, &*self.mapper);

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

                let valid_actions_with_policies = self.mapper.policy_to_valid_actions(game_state, policy_scores);

                GameStateAnalysis {
                    policy_scores: valid_actions_with_policies,
                    value_score
                }
            })
            .collect();

        Ok(result)
    }
}

pub struct GameStateAnalysisFuture<S,A,E,Map> {
    game_state: S,
    id: usize,
    batching_model: Arc<BatchingModel<S,A,E,Map>>
}

impl<S,A,E,Map> GameStateAnalysisFuture<S,A,E,Map> {
    fn new(
        game_state: S,
        id: usize,
        batching_model: Arc<BatchingModel<S,A,E,Map>>
    ) -> Self
    {
        Self { game_state, id, batching_model }
    }
}

impl<S,A,E,Map> Future for GameStateAnalysisFuture<S,A,E,Map>
where
    S: Clone + PartialEq + Hash,
    A: Clone,
    E: GameEngine<State=S,Action=A> + Send + Sync + 'static,
    Map: Mapper<S,A>
{
    type Output = GameStateAnalysis<A>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        (*self).batching_model.poll(self.id, &self.game_state, cx.waker())
    }
}

#[derive(Serialize)]
struct RequestImage {
    input_image: Vec<Vec<Vec<f32>>>
}

#[derive(Debug, Deserialize)]
struct PredictionResults {
    predictions: Vec<HashMap<String,Vec<f32>>>
}

fn get_model_url(model_info: &ModelInfo) -> String {
    format!(
        "http://localhost:8501/v1/models/exported_models/versions/{version}:predict",
        version = model_info.get_run_num()
    )
}

fn game_states_to_request_body<S,A,Map>(game_states: &Vec<&S>, mapper: &Map) -> serde_json::value::Value
where
    Map: Mapper<S,A>
{
    let game_states: Vec<_> = game_states.iter().map(|game_state| RequestImage {
        input_image: mapper.game_state_to_input(game_state)
    }).collect();

    json!({
        "instances": game_states
    })
}
