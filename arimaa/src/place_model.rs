use crate::{place_mappings::Mapper, PlaceTranspositionEntry};
use std::path::PathBuf;

use super::engine::Engine;
use super::game_state::GameState;
use super::value::Value;
use arimaa_engine::Action;
use model::model_info::ModelInfo;
use tensorflow_model::{TensorflowModel, TensorflowModelOptions};

use anyhow::Result;

/*
    Layers:
    In:
    6 piece boards
    6 curr pieces remaining temp board
    1 player
    1 piece placement bit
    Out:
    6 pieces
*/

#[derive(Default)]
pub struct ModelFactory {}

impl ModelFactory {
    pub fn new() -> Self {
        Self {}
    }
}

impl ModelFactory {
    pub fn load(
        &self,
        model_dir: PathBuf,
        model_options: TensorflowModelOptions,
        model_info: ModelInfo,
    ) -> Result<TensorflowModel<GameState, Action, Value, Engine, Mapper, PlaceTranspositionEntry>>
    {
        let mapper = Mapper::new();

        let table_size = std::env::var("PLACE_TABLE_SIZE")
            .map(|v| {
                v.parse::<usize>()
                    .expect("PLACE_TABLE_SIZE must be a valid number")
            })
            .unwrap_or(800);

        TensorflowModel::load(
            model_dir,
            model_options,
            model_info,
            Engine::new(),
            mapper,
            table_size,
        )
    }
}
