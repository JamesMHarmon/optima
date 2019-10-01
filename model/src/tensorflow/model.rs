use std::fmt::Formatter;
use std::fmt::Display;
use std::hash::Hash;
use std::collections::HashMap;
use std::fs::{self};
use std::future::Future;
use std::pin::Pin;
use std::process::{Command,Stdio};
use std::sync::{Arc,Mutex,atomic::{AtomicBool,AtomicUsize,Ordering}};
use std::task::{Context,Poll,Waker};
use std::time::Instant;
use chrono::{Utc};
use crossbeam_queue::{SegQueue};
use failure::Error;
use itertools::{izip,Itertools};
use tensorflow::{Graph,Operation,Session,SessionOptions,SessionRunArgs,Tensor};


use engine::game_state::GameState;
use engine::engine::GameEngine;

use super::constants::{ANALYSIS_REQUEST_BATCH_SIZE,ANALYSIS_REQUEST_THREADS,TRAIN_DATA_CHUNK_SIZE};
use super::paths::Paths;
use super::super::analytics::{self,ActionWithPolicy,GameStateAnalysis};
use super::super::model::{Model as ModelTrait,TrainOptions};
use super::super::model_info::ModelInfo;
use super::super::node_metrics::NodeMetrics;
use super::super::position_metrics::PositionMetrics;

#[cfg_attr(feature="tensorflow_system_alloc", global_allocator)]
#[cfg(feature="tensorflow_system_alloc")]
static ALLOCATOR: std::alloc::System = std::alloc::System;

pub struct TensorflowModel<S,A,E,Map>
{
    model_info: ModelInfo,
    batching_model: Arc<BatchingModel<S,A,E,Map>>,
    alive: Arc<AtomicBool>,
    id_generator: Arc<AtomicUsize>,
    mapper: Arc<Map>
}

pub trait Mapper<S,A> {
    fn game_state_to_input(&self, game_state: &S) -> Vec<f32>;
    fn get_input_dimensions(&self) -> [u64; 3];
    fn policy_metrics_to_expected_output(&self, game_state: &S, policy: &NodeMetrics<A>) -> Vec<f32>;
    fn policy_to_valid_actions(&self, game_state: &S, policy_scores: &[f32]) -> Vec<ActionWithPolicy<A>>;
}

impl<S,A,E,Map> TensorflowModel<S,A,E,Map>
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

impl<S,A,E,Map> ModelTrait for TensorflowModel<S,A,E,Map>
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

pub struct SessionAndOps {
    session: Session,
    op_input: Operation,
    op_value_head: Operation,
    op_policy_head: Operation
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

    let source_paths = Paths::from_model_info(&source_model_info);
    let source_base_path = source_paths.get_base_path();

    let mut train_data_file_names = vec!();

    for (i, sample_metrics) in sample_metrics.chunks(TRAIN_DATA_CHUNK_SIZE).into_iter().enumerate() {
        let sample_metrics: Vec<_> = sample_metrics.collect();

        let dimensions = mapper.get_input_dimensions();
        let X: Vec<_> = sample_metrics.iter().map(|v| {
            let X: Vec<_> = mapper.game_state_to_input(&v.game_state);
            let X: Vec<_> = X.chunks_exact(dimensions[2] as usize).map(|v| NumVec(v.to_owned())).collect();
            let X: Vec<_> = X.chunks_exact(dimensions[1] as usize).map(|v| NumVec(v.to_owned())).collect();
            NumVec(X)
        }).collect();

        let yv: Vec<_> = sample_metrics.iter().map(|v| v.score).collect();
        let yp: Vec<_> = sample_metrics.iter().map(|v| NumVec(mapper.policy_metrics_to_expected_output(&v.game_state, &v.policy))).collect();

        // Note that we are no longer using serde_json here due to the way that it elongates floats.
        // Perhaps there is a way to override the way that serde_json formats floats?
        let json = format!("{{\"x\":{},\"yv\":{},\"yp\":{}}}", NumVec(X), NumVec(yv), NumVec(yp));

        let train_data_file_name = format!("training_data_{}.json", i);
        let train_data_path = source_base_path.join(&train_data_file_name);

        println!("Writing data to {:?}", &train_data_path);

        fs::write(train_data_path, json)?;

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

    for file_name in train_data_file_names {
        let path = source_base_path.join(file_name);
        fs::remove_file(path)?;
    }

    let docker_cmd = format!("docker run --rm \
        --runtime=nvidia \
        --mount type=bind,source=\"$(pwd)/{game_name}_runs\",target=/{game_name}_runs \
        tensorflow/tensorflow:latest-gpu \
        usr/local/bin/saved_model_cli convert \
        --dir /{game_name}_runs/{run_name}/exported_models/{source_run_num} \
        --output_dir /{game_name}_runs/{run_name}/tensorrt_models/{target_run_num} \
        --tag_set serve \
        tensorrt \
        --precision_mode FP16 \
        --max_batch_size 2048 \
        --is_dynamic_op False",
        game_name = source_model_info.get_game_name(),
        run_name = source_model_info.get_run_name(),
        source_run_num = source_model_info.get_run_num(),
        target_run_num = target_model_info.get_run_num()
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

    let docker_cmd = format!("docker run --rm \
        --runtime=nvidia \
        --mount type=bind,source=\"$(pwd)/{game_name}_runs\",target=/{game_name}_runs \
        tensorflow/tensorflow:latest-gpu \
        usr/local/bin/saved_model_cli convert \
        --dir /{game_name}_runs/{run_name}/exported_models/1 \
        --output_dir /{game_name}_runs/{run_name}/tensorrt_models/1 \
        --tag_set serve \
        tensorrt \
        --precision_mode FP16 \
        --max_batch_size 2048 \
        --is_dynamic_op False",
        game_name = game_name,
        run_name = run_name
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

impl<S,A,E,Map> Drop for TensorflowModel<S,A,E,Map> {
    fn drop(&mut self) {
        self.alive.store(false, Ordering::SeqCst);
    }
}

pub struct BatchingModel<S,A,E,Map> {
    states_to_analyse: SegQueue<(usize, S, Waker)>,
    states_analysed: Mutex<HashMap<usize, GameStateAnalysis<A>>>,
    num_nodes_analysed: AtomicUsize,
    min_batch_size: AtomicUsize,
    max_batch_size: AtomicUsize,
    engine: E,
    mapper: Arc<Map>,
    session: Mutex<SessionAndOps>
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
        let states_analysed = Mutex::new(HashMap::with_capacity(ANALYSIS_REQUEST_BATCH_SIZE * ANALYSIS_REQUEST_THREADS));
        let num_nodes_analysed = AtomicUsize::new(0);
        let min_batch_size = AtomicUsize::new(std::usize::MAX);
        let max_batch_size = AtomicUsize::new(0);

        let mut graph = Graph::new();

        let exported_model_path = format!(
            "{game_name}_runs/{run_name}/tensorrt_models/{run_num}",
            game_name = model_info.get_game_name(),
            run_name = model_info.get_run_name(),
            run_num = model_info.get_run_num(),
        );

        let exported_model_path = std::env::current_dir().unwrap().join(exported_model_path);

        println!("{:?}", exported_model_path);

        let session = Session::from_saved_model(
            &SessionOptions::new(),
            &["serve"],
            &mut graph,
            exported_model_path
        ).unwrap();

        let op_input = graph.operation_by_name_required("input_1").unwrap();
        let op_value_head = graph.operation_by_name_required("value_head/Tanh").unwrap();
        let op_policy_head = graph.operation_by_name_required("policy_head/Softmax").unwrap();

        Self {
            states_to_analyse,
            states_analysed,
            num_nodes_analysed,
            min_batch_size,
            max_batch_size,
            engine,
            mapper,
            session: Mutex::new(SessionAndOps {
                session,
                op_input,
                op_value_head,
                op_policy_head
            })
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

        let states_analysed_map = &mut *self.states_analysed.lock().unwrap();
        for ((id, _s, waker), analysis) in states_to_analyse.into_iter().zip(analysis) {
            states_analysed_map.insert(id, analysis);
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
        let mapper = &*self.mapper;

        let inputs: Vec<_> = game_states.iter().map(|game_state| mapper.game_state_to_input(game_state)).collect();

        let batch_size = game_states.len();
        let input_dim = mapper.get_input_dimensions();
        let input_dimensions = [batch_size as u64, input_dim[0], input_dim[1], input_dim[2]];
        let flattened_inputs: Vec<f32> = inputs.into_iter().flatten().collect();
        let mut value_head_outputs = Vec::with_capacity(batch_size);
        let mut policy_head_outputs = Vec::with_capacity(batch_size);
        let policy_dimension;

        {
            let input_tensor = Tensor::new(&input_dimensions).with_values(&flattened_inputs).unwrap();
            let session = self.session.lock().unwrap();

            let mut output_step = SessionRunArgs::new();
            output_step.add_feed(&session.op_input, 0, &input_tensor);
            let value_head_fetch_token = output_step.request_fetch(&session.op_value_head, 0);
            let policy_head_fetch_token = output_step.request_fetch(&session.op_policy_head, 0);

            session.session.run(&mut output_step).unwrap();

            let value_head_output: Tensor<f32> = output_step.fetch(value_head_fetch_token).unwrap();
            let policy_head_output: Tensor<f32> = output_step.fetch(policy_head_fetch_token).unwrap();

            policy_dimension = policy_head_output.dims()[1];

            value_head_outputs.extend(value_head_output.into_iter().map(|v| *v));
            policy_head_outputs.extend(policy_head_output.into_iter().map(|c| *c));
        }
        
        let analysis_results = izip!(
            game_states.into_iter(),
            value_head_outputs.into_iter(),
            policy_head_outputs.chunks_exact(policy_dimension as usize).into_iter()
        ).map(|(game_state, value_score, policy_scores)| {
                let valid_actions_with_policies = self.mapper.policy_to_valid_actions(
                    game_state,
                    &policy_scores
                );

                GameStateAnalysis {
                    policy_scores: valid_actions_with_policies,
                    value_score: value_score as f32
                }
            })
            .collect();

        Ok(analysis_results)
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

#[derive(Clone)]
struct NumVec<T>(Vec<T>);

impl<T> Display for NumVec<T>
    where T: Display
{
    fn fmt(&self, f: &mut Formatter) -> Result<(), std::fmt::Error> {
        let mut comma_separated = String::new();

        for num in &self.0[0..self.0.len() - 1] {
            comma_separated.push_str(&num.to_string());
            comma_separated.push_str(",");
        }

        comma_separated.push_str(&self.0[self.0.len() - 1].to_string());
        write!(f, "[{}]", comma_separated)
    }
}