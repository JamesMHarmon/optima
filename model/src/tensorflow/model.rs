use std::fmt::Formatter;
use std::fmt::Display;
use std::hash::Hash;
use std::fs::{self};
use std::future::Future;
use std::pin::Pin;
use std::process::{Command,Stdio};
use std::sync::{Arc,atomic::{AtomicBool,AtomicUsize,Ordering}};
use std::task::{Context,Poll,Waker};
use std::time::Instant;
use chrono::{Utc};
use crossbeam_queue::{SegQueue};
use failure::Error;
use itertools::{izip,Itertools};
use tensorflow::{Graph,Operation,Session,SessionOptions,SessionRunArgs,Tensor};

use common::incrementing_map::IncrementingMap;
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

pub struct TensorflowModel<E,Map>
{
    model_info: ModelInfo,
    batching_model: Arc<BatchingModel<E,Map>>,
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

impl<S,A,E,Map> TensorflowModel<E,Map>
where
    S: PartialEq + Hash + Send + Sync + 'static,
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
        let batching_model = Arc::new(BatchingModel::new(engine, mapper.clone()));
        let alive = Arc::new(AtomicBool::new(true));

        for i in 0..ANALYSIS_REQUEST_THREADS {
            let batching_model_ref = batching_model.clone();
            let alive_ref = alive.clone();
            let model_info = model_info.clone();
            let mapper = mapper.clone();

            std::thread::spawn(move || {
                let mut last_report = Instant::now();
                let input_dim = mapper.get_input_dimensions();
                let predictor = Predictor::new(&model_info, input_dim);

                loop {
                    let states_to_analyse = batching_model_ref.get_states_to_analyse();
                    let game_states_to_predict: Vec<&Vec<f32>> = states_to_analyse.iter().map(|(_,input,_)| input).collect();

                    if states_to_analyse.len() == 0 {
                        std::thread::sleep(std::time::Duration::from_millis(1));
                    }

                    let predictions = predictor.predict(game_states_to_predict).unwrap();
                    let predictions: Vec<_> = predictions.into_iter()
                        .zip(states_to_analyse.into_iter())
                        .map(|(prediction,(id,_,waker))| (id, prediction, waker))
                        .collect();

                    batching_model_ref.provide_analysis(predictions);

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

impl<S,A,E,Map> ModelTrait for TensorflowModel<E,Map>
where
    S: GameState + Send + Sync + Unpin + 'static,
    A: Clone + Send + Sync + 'static,
    E: GameEngine<State=S,Action=A> + Send + Sync + 'static,
    Map: Mapper<S,A> + Send + Sync + 'static
{
    type State = S;
    type Action = A;
    type Analyzer = GameAnalyzer<E,Map>;

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

struct Predictor {
    session: SessionAndOps,
    input_dimensions: [u64; 3]
}

impl Predictor {
    fn new(model_info: &ModelInfo, input_dimensions: [u64; 3]) -> Self {
        let mut graph = Graph::new();

        let exported_model_path = format!(
            "{game_name}_runs/{run_name}/tensorrt_models/{model_num}",
            game_name = model_info.get_game_name(),
            run_name = model_info.get_run_name(),
            model_num = model_info.get_model_num(),
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
            input_dimensions,
            session: SessionAndOps {
                session,
                op_input,
                op_value_head,
                op_policy_head
            }
        }
    }

    fn predict(&self, game_state_inputs: Vec<&Vec<f32>>) -> Result<Vec<(Vec<f32>, f32)>, Error> {
        let batch_size = game_state_inputs.len();

        if batch_size == 0 {
            return Ok(Vec::new());
        }

        let input_dim = self.input_dimensions;
        let input_dimensions = [batch_size as u64, input_dim[0], input_dim[1], input_dim[2]];
        let flattened_inputs: Vec<f32> = game_state_inputs.iter().map(|v| v.iter()).flatten().map(|v| *v).collect();
        let mut value_head_outputs = Vec::with_capacity(batch_size);
        let mut policy_head_outputs = Vec::with_capacity(batch_size);
        let policy_dimension;

        let input_tensor = Tensor::new(&input_dimensions).with_values(&flattened_inputs).unwrap();
        let session = &self.session;

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
        
        let analysis_results: Vec<_> = izip!(
            policy_head_outputs.chunks_exact(policy_dimension as usize).map(|c| c.to_vec()),
            value_head_outputs
        ).collect();

        Ok(analysis_results)
    }
}

pub struct SessionAndOps {
    session: Session,
    op_input: Operation,
    op_value_head: Operation,
    op_policy_head: Operation
}

pub struct GameAnalyzer<E,Map> {
    batching_model: Arc<BatchingModel<E,Map>>,
    id_generator: Arc<AtomicUsize>
}

impl<S,A,E,Map> analytics::GameAnalyzer for GameAnalyzer<E,Map>
where
    S: Clone + PartialEq + Hash + Unpin,
    A: Clone,
    E: GameEngine<State=S,Action=A> + Send + Sync + 'static,
    Map: Mapper<S,A>
{
    type State = S;
    type Action = A;
    type Future = GameStateAnalysisFuture<S,E,Map>;

    /// Outputs a value from [-1, 1] depending on the player to move's evaluation of the current state.
    /// If the evaluation is a draw then 0.0 will be returned.
    /// Along with the value output a list of policy scores for all VALID moves is returned. If the position
    /// is terminal then the vector will be empty.
    fn get_state_analysis(&self, game_state: &S) -> GameStateAnalysisFuture<S,E,Map> {
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
        -e SOURCE_MODEL_PATH=/{game_name}_runs/{run_name}/models/{game_name}_{run_name}_{source_model_num:0>5}.h5 \
        -e TARGET_MODEL_PATH=/{game_name}_runs/{run_name}/models/{game_name}_{run_name}_{target_model_num:0>5}.h5 \
        -e EXPORT_MODEL_PATH=/{game_name}_runs/{run_name}/exported_models/{target_model_num} \
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
        source_model_num = source_model_info.get_model_num(),
        target_model_num = target_model_info.get_model_num(),
        train_ratio = options.train_ratio,
        train_batch_size = options.train_batch_size,
        epochs = (source_model_info.get_model_num() - 1) + options.epochs,
        initial_epoch = (source_model_info.get_model_num() - 1),
        train_data_paths = train_data_paths.map(|p| format!("\"{}\"", p)).join(","),
        learning_rate = options.learning_rate,
        policy_loss_weight = options.policy_loss_weight,
        value_loss_weight = options.value_loss_weight
    );

    run_cmd(&docker_cmd)?;

    for file_name in train_data_file_names {
        let path = source_base_path.join(file_name);
        fs::remove_file(path)?;
    }

    create_tensorrt_model(source_model_info.get_game_name(), source_model_info.get_run_name(), target_model_info.get_model_num())?;

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

    run_cmd(&docker_cmd)?;

    create_tensorrt_model(game_name, run_name, 1)?;

    println!("Model creation process complete");

    Ok(())
}

fn create_tensorrt_model(game_name: &str, run_name: &str, model_num: usize) -> Result<(), Error> {
    let docker_cmd = format!("docker run --rm \
        --runtime=nvidia \
        --mount type=bind,source=\"$(pwd)/{game_name}_runs\",target=/{game_name}_runs \
        tensorflow/tensorflow:latest-gpu \
        usr/local/bin/saved_model_cli convert \
        --dir /{game_name}_runs/{run_name}/exported_models/{model_num} \
        --output_dir /{game_name}_runs/{run_name}/tensorrt_models/{model_num} \
        --tag_set serve \
        tensorrt \
        --precision_mode FP16 \
        --max_batch_size 512 \
        --is_dynamic_op False",
        game_name = game_name,
        run_name = run_name,
        model_num = model_num
    );

    run_cmd(&docker_cmd)?;

    Ok(())
}

fn run_cmd(cmd: &str) -> Result<(), Error> {
    println!("\n");
    println!("{}", cmd);
    println!("\n");

    let mut cmd = Command::new("/bin/bash")
        .arg("-c")
        .arg(cmd)
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()?;

    let result = cmd.wait();

    println!("OUTPUT: {:?}", result);

    Ok(())
}

impl<E,Map> Drop for TensorflowModel<E,Map> {
    fn drop(&mut self) {
        self.alive.store(false, Ordering::SeqCst);
    }
}

pub struct BatchingModel<E,Map> {
    states_to_analyse: SegQueue<(usize, Vec<f32>, Waker)>,
    states_analysed: IncrementingMap<(Vec<f32>,f32)>,
    num_nodes_analysed: AtomicUsize,
    min_batch_size: AtomicUsize,
    max_batch_size: AtomicUsize,
    engine: E,
    mapper: Arc<Map>
}

impl<S,A,E,Map> BatchingModel<E,Map>
where
    S: PartialEq + Hash,
    A: Clone,
    E: GameEngine<State=S,Action=A> + Send + Sync + 'static,
    Map: Mapper<S,A>
{
    fn new(engine: E, mapper: Arc<Map>) -> Self
    {
        let states_to_analyse = SegQueue::new();
        let states_analysed = IncrementingMap::with_capacity(ANALYSIS_REQUEST_BATCH_SIZE * ANALYSIS_REQUEST_THREADS);
        let num_nodes_analysed = AtomicUsize::new(0);
        let min_batch_size = AtomicUsize::new(std::usize::MAX);
        let max_batch_size = AtomicUsize::new(0);

        Self {
            states_to_analyse,
            states_analysed,
            num_nodes_analysed,
            min_batch_size,
            max_batch_size,
            engine,
            mapper
        }
    }

    fn get_states_to_analyse(&self) -> Vec<(usize, Vec<f32>, Waker)> {
        let states_to_analyse_queue = &self.states_to_analyse;

        let mut states_to_analyse: Vec<_> = Vec::with_capacity(ANALYSIS_REQUEST_BATCH_SIZE);
        while let Ok(state_to_analyse) = states_to_analyse_queue.pop() {
            states_to_analyse.push(state_to_analyse);

            if states_to_analyse.len() >= ANALYSIS_REQUEST_BATCH_SIZE {
                break;
            }
        }

        states_to_analyse
    }

    fn provide_analysis(&self, analysis: Vec<(usize, (Vec<f32>,f32), Waker)>) {
        let analysis_len = analysis.len();
        self.min_batch_size.fetch_min(analysis_len, Ordering::SeqCst);
        self.max_batch_size.fetch_max(analysis_len, Ordering::SeqCst);
        let mut wakers = Vec::with_capacity(analysis.len());

        for (id, analysis, waker) in analysis.into_iter() {
            self.states_analysed.insert(id, analysis);
            wakers.push(waker);
        }

        for waker in wakers.into_iter() {
            waker.wake();
        }
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

    fn poll(&self, id: usize) -> Poll<(Vec<f32>,f32)> {
        let analysis = self.states_analysed.remove(id);

        match analysis {
            Some(analysis) => {
                self.num_nodes_analysed.fetch_add(1, Ordering::SeqCst);
                Poll::Ready(analysis)
            },
            None => Poll::Pending
        }
    }

    fn request(&self, id: usize, game_state: &S, waker: &Waker) -> Poll<GameStateAnalysis<A>> {
        if let Some(value) = self.engine.is_terminal_state(game_state) {
            self.num_nodes_analysed.fetch_add(1, Ordering::SeqCst);
            Poll::Ready(GameStateAnalysis::new(
                value,
                Vec::new()
            ))
        } else {
            self.states_to_analyse.push((
                id,
                self.mapper.game_state_to_input(game_state),
                waker.clone()
            ));

            Poll::Pending
        }
    }
}

pub struct GameStateAnalysisFuture<S,E,Map> {
    game_state: S,
    id: usize,
    has_requested: bool,
    batching_model: Arc<BatchingModel<E,Map>>
}

impl<S,E,Map> GameStateAnalysisFuture<S,E,Map> {
    fn new(
        game_state: S,
        id: usize,
        batching_model: Arc<BatchingModel<E,Map>>
    ) -> Self
    {
        Self { game_state, id, batching_model, has_requested: false }
    }
}

impl<S,A,E,Map> Future for GameStateAnalysisFuture<S,E,Map>
where
    S: PartialEq + Hash + Unpin,
    A: Clone,
    E: GameEngine<State=S,Action=A> + Send + Sync + 'static,
    Map: Mapper<S,A>
{
    type Output = GameStateAnalysis<A>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if !self.has_requested {
            let self_mut = self.get_mut();
            self_mut.has_requested = true;
            self_mut.batching_model.request(self_mut.id, &self_mut.game_state, cx.waker())
        } else {
            match (*self).batching_model.poll(self.id) {
                Poll::Ready((policy_scores, value_score)) => {
                    let valid_actions_with_policies = self.batching_model.mapper.policy_to_valid_actions(
                        &self.game_state,
                        &policy_scores
                    );

                    Poll::Ready(GameStateAnalysis {
                        policy_scores: valid_actions_with_policies,
                        value_score: value_score as f32
                    })
                },
                Poll::Pending => Poll::Pending
            }
        }
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