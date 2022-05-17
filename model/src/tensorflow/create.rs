use anyhow::Result;
use log::info;
use std::fs::{self};

use super::super::model_info::ModelInfo;
use super::get_model_dir;
use super::model_options::write_options;
use super::{run_cmd, TensorflowModelOptions};

#[allow(non_snake_case)]
pub fn create(model_info: &ModelInfo, options: &TensorflowModelOptions) -> Result<()> {
    let game_name = model_info.get_game_name();
    let run_name = model_info.get_run_name();

    let model_dir = get_model_dir(model_info);
    fs::create_dir_all(model_dir)?;

    write_options(model_info, options)?;

    let docker_cmd = format!(
        "docker run --rm \
        --gpus all \
        --mount type=bind,source=\"$(pwd)/{game_name}_runs\",target=/{game_name}_runs \
        -e TARGET_MODEL_PATH=/{game_name}_runs/{run_name}/models/{game_name}_{run_name}_00001.h5 \
        -e EXPORT_MODEL_PATH=/{game_name}_runs/{run_name}/exported_models/1 \
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
        quoridor_engine/create:latest",
        game_name = game_name,
        run_name = run_name,
        input_h = options.channel_height,
        input_w = options.channel_width,
        input_c = options.channels,
        output_size = options.output_size,
        moves_left_size = options.moves_left_size,
        num_filters = options.num_filters,
        num_blocks = options.num_blocks,
    );

    run_cmd(&docker_cmd)?;

    info!("Model creation process complete");

    Ok(())
}
