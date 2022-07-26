use arimaa_engine::Action;
use model::ModelInfo;
use tensorflow_model::Archive as ArchiveModel;
use tensorflow_model::{GameAnalyzer, TensorflowModel};

use super::engine::Engine;
use super::game_state::GameState;
use super::mappings::Mapper;
use super::value::Value;
use super::TranspositionEntry;

pub type Analyzer = GameAnalyzer<GameState, Action, Value, Engine, Mapper, TranspositionEntry>;

pub struct Model(
    ArchiveModel<TensorflowModel<GameState, Action, Value, Engine, Mapper, TranspositionEntry>>,
);

impl Model {
    pub fn new(
        model: ArchiveModel<
            TensorflowModel<GameState, Action, Value, Engine, Mapper, TranspositionEntry>,
        >,
    ) -> Self {
        Self(model)
    }
}

impl model::Analyzer for Model {
    type State = GameState;
    type Action = Action;
    type Value = Value;
    type Analyzer = Analyzer;

    fn analyzer(&self) -> Self::Analyzer {
        self.0.inner().analyzer()
    }
}

impl model::Info for Model {
    fn info(&self) -> &ModelInfo {
        self.0.inner().info()
    }
}
