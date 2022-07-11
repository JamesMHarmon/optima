use anyhow::Result;
use std::path::PathBuf;

use super::engine::Engine;
use super::game_state::GameState;
use super::play_mappings::Mapper;
use super::value::Value;
use super::PlayTranspositionEntry;
use arimaa_engine::Action;
use model::model_info::ModelInfo;
use tensorflow_model::TensorflowModelOptions;

/*
    Layers:
    In:
    6 curr piece boards
    6 opp piece boards
    3 current step
    1 banned pieces board
    1 trap squares

    Out:
    40 directional boards (substract irrelevant squares)
    12 push pull boards
    1 pass bit
*/

#[derive(Default)]
pub struct ModelFactory {}

impl ModelFactory {
    pub fn new() -> Self {
        Self {}
    }
}

#[cfg(feature = "all")]
impl ModelFactory {
    pub fn load(
        &self,
        model_dir: PathBuf,
        model_options: TensorflowModelOptions,
        model_info: ModelInfo,
    ) -> Result<
        tensorflow_model::TensorflowModel<
            GameState,
            Action,
            Value,
            Engine,
            Mapper,
            PlayTranspositionEntry,
        >,
    > {
        let mapper = Mapper::new();

        let table_size = std::env::var("PLAY_TABLE_SIZE")
            .map(|v| {
                v.parse::<usize>()
                    .expect("PLAY_TABLE_SIZE must be a valid number")
            })
            .unwrap_or(0);

        tensorflow_model::TensorflowModel::load(
            model_dir,
            model_options,
            model_info,
            Engine::new(),
            mapper,
            table_size,
        )
    }
}
