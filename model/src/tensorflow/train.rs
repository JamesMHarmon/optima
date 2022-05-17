use anyhow::{anyhow, Result};
use engine::game_state::GameState;
use half::f16;
use itertools::Itertools;
use log::info;
use rand::seq::SliceRandom;
use std::fs::{self};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use super::super::model::TrainOptions;
use super::super::model_info::ModelInfo;
use super::super::position_metrics::PositionMetrics;
use super::model_options::get_options;
use super::*;

pub fn get_model_dir(model_info: &ModelInfo) -> PathBuf {
    Paths::from_model_info(model_info).get_models_path()
}

#[allow(non_snake_case)]
pub fn train<S, A, V, I, Map, Te>(
    source_model_info: &ModelInfo,
    target_model_info: &ModelInfo,
    sample_metrics: I,
    mapper: Arc<Map>,
    options: &TrainOptions,
) -> Result<()>
where
    S: GameState + Send + Sync + 'static,
    A: Clone + Send + Sync + 'static,
    V: Send + 'static,
    I: Iterator<Item = PositionMetrics<S, A, V>>,
    Map: Mapper<S, A, V, Te> + Send + Sync + 'static,
{
    info!(
        "Training from {} to {}",
        source_model_info.get_model_name(),
        target_model_info.get_model_name()
    );

    let model_options = get_options(source_model_info)?;
    let moves_left_size = model_options.moves_left_size;
    let train_batch_size = options.train_batch_size;
    let source_paths = Paths::from_model_info(source_model_info);
    let source_base_path = source_paths.get_base_path();

    let mut train_data_file_names = vec![];
    let mut handles = vec![];

    let mut train_data_chunk_size = std::env::var("TRAIN_DATA_CHUNK_SIZE")
        .map(|v| {
            v.parse::<usize>()
                .expect("TRAIN_DATA_CHUNK_SIZE must be a valid int")
        })
        .unwrap_or(TRAIN_DATA_CHUNK_SIZE);

    if train_data_chunk_size & train_batch_size != 0 {
        let train_data_chunk_size_before_round = train_data_chunk_size;
        train_data_chunk_size = (train_data_chunk_size / train_batch_size) * train_batch_size;
        info!(
            "Reduced train_data_chunk_size from {} to {} be a multiple of batch size",
            train_data_chunk_size_before_round, train_data_chunk_size
        );
    }

    let total_samples = Arc::new(AtomicUsize::new(0));

    for (i, sample_metrics) in sample_metrics
        .chunks(train_data_chunk_size)
        .into_iter()
        .enumerate()
    {
        let train_data_file_name = format!("training_data_{}.npy", i);
        let train_data_path = source_base_path.join(&train_data_file_name);
        info!("Writing data to {:?}", &train_data_path);
        train_data_file_names.push(train_data_file_name);
        let sample_metrics_chunk = sample_metrics.collect::<Vec<_>>();
        let total_samples = total_samples.clone();
        let mapper = mapper.clone();

        handles.push(std::thread::spawn(move || {
            let rng = &mut rand::thread_rng();
            let mut wtr = npy::OutFile::open(train_data_path).unwrap();

            let mut train_data_by_batch_size = sample_metrics_chunk.len();
            if sample_metrics_chunk.len() % train_batch_size != 0 {
                train_data_by_batch_size = (sample_metrics_chunk.len() / train_batch_size) * train_batch_size;
                info!(
                    "Reduced train_data_by_batch_size from {} to {} to be a multiple of batch size",
                    sample_metrics_chunk.len(),
                    train_data_by_batch_size
                );
            }

            total_samples.fetch_add(train_data_by_batch_size, Ordering::SeqCst);

            for metric in sample_metrics_chunk
                .into_iter()
                .take(train_data_by_batch_size)
            {
                let metric_symmetires = mapper.get_symmetries(metric);
                let metric = metric_symmetires
                    .choose(rng)
                    .expect("Expected at least one metric to return from symmetries.");

                let policy_output =
                    mapper.policy_metrics_to_expected_output(&metric.game_state, &metric.policy);
                let value_output =
                    mapper.map_value_to_value_output(&metric.game_state, &metric.score);
                let moves_left_output =
                    map_moves_left_to_one_hot(metric.moves_left, moves_left_size);

                let sum_of_policy = policy_output.iter().filter(|&&x| x >= 0.0).sum::<f32>();
                assert!(
                    f32::abs(sum_of_policy - 1.0) <= f32::EPSILON * policy_output.len() as f32,
                    "Policy output should sum to 1.0 but actual sum is {}",
                    sum_of_policy
                );

                for record in mapper
                    .game_state_to_input(&metric.game_state, Mode::Train)
                    .into_iter()
                    .map(f16::to_f32)
                    .chain(
                        policy_output
                            .into_iter()
                            .chain(std::iter::once(value_output).chain(moves_left_output)),
                    )
                {
                    wtr.push(&record).unwrap();
                }
            }

            wtr.close().unwrap();
        }));
    }

    for handle in handles {
        handle
            .join()
            .map_err(|_| anyhow!("Thread failed to write training data"))?;
    }

    let train_data_paths = train_data_file_names.iter().map(|file_name| {
        format!(
            "/{game_name}_runs/{run_name}/{file_name}",
            game_name = source_model_info.get_game_name(),
            run_name = source_model_info.get_run_name(),
            file_name = file_name
        )
    });

    let docker_cmd = format!("docker run --rm \
        --gpus all \
        --mount type=bind,source=\"$(pwd)/{game_name}_runs\",target=/{game_name}_runs \
        -e SOURCE_MODEL_PATH=/{game_name}_runs/{run_name}/models/{game_name}_{run_name}_{source_model_num:0>5}.h5 \
        -e TARGET_MODEL_PATH=/{game_name}_runs/{run_name}/models/{game_name}_{run_name}_{target_model_num:0>5}.h5 \
        -e EXPORT_MODEL_PATH=/{game_name}_runs/{run_name}/exported_models/{target_model_num} \
        -e TENSOR_BOARD_PATH=/{game_name}_runs/{run_name}/tensorboard \
        -e TRAIN_STATE_PATH=/{game_name}_runs/{run_name}/train_state.json \
        -e EPOCH={initial_epoch} \
        -e DATA_PATHS={train_data_paths} \
        -e TOTAL_SAMPLES={total_samples} \
        -e TRAIN_RATIO={train_ratio} \
        -e TRAIN_BATCH_SIZE={train_batch_size} \
        -e MAX_GRAD_NORM={max_grad_norm} \
        -e LEARNING_RATE={learning_rate} \
        -e POLICY_LOSS_WEIGHT={policy_loss_weight} \
        -e VALUE_LOSS_WEIGHT={value_loss_weight} \
        -e MOVES_LEFT_LOSS_WEIGHT={moves_left_loss_weight} \
        -e INPUT_H={input_h} \
        -e INPUT_W={input_w} \
        -e INPUT_C={input_c} \
        -e OUTPUT_SIZE={output_size} \
        -e MOVES_LEFT_SIZE={moves_left_size} \
        -e NUM_FILTERS={num_filters} \
        -e NUM_BLOCKS={num_blocks} \
        -e NVIDIA_VISIBLE_DEVICES=0 \
        -e CUDA_VISIBLE_DEVICES=0 \
        -e TF_FORCE_GPU_ALLOW_GROWTH=true \
        quoridor_engine/train:latest",
        game_name = source_model_info.get_game_name(),
        run_name = source_model_info.get_run_name(),
        source_model_num = source_model_info.get_model_num(),
        target_model_num = target_model_info.get_model_num(),
        train_ratio = options.train_ratio,
        train_batch_size = options.train_batch_size,
        initial_epoch = source_model_info.get_model_num(),
        train_data_paths = train_data_paths.map(|p| format!("\"{}\"", p)).join(","),
        total_samples = total_samples.load(Ordering::SeqCst),
        learning_rate = options.learning_rate,
        max_grad_norm = options.max_grad_norm,
        policy_loss_weight = options.policy_loss_weight,
        value_loss_weight = options.value_loss_weight,
        moves_left_loss_weight = options.moves_left_loss_weight,
        input_h = model_options.channel_height,
        input_w = model_options.channel_width,
        input_c = model_options.channels,
        output_size = model_options.output_size,
        moves_left_size = model_options.moves_left_size,
        num_filters = model_options.num_filters,
        num_blocks = model_options.num_blocks,
    );

    run_cmd(&docker_cmd)?;

    for file_name in train_data_file_names {
        let path = source_base_path.join(file_name);
        fs::remove_file(path)?;
    }

    info!("Training process complete");

    Ok(())
}

fn map_moves_left_to_one_hot(moves_left: usize, moves_left_size: usize) -> Vec<f32> {
    if moves_left_size == 0 {
        return vec![];
    }

    let moves_left = moves_left.max(0).min(moves_left_size);
    let mut moves_left_one_hot = vec![0f32; moves_left_size];
    moves_left_one_hot[moves_left - 1] = 1.0;

    moves_left_one_hot
}
