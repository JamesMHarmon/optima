use arimaa_engine::Action;
use model::ModelInfo;
use tensorflow_model::Archive as ArchiveModel;
use tensorflow_model::{GameAnalyzer, TensorflowModel};

use super::engine::Engine;
use super::game_state::GameState;
use super::mappings::Mapper;
use super::value::Value;
use super::TranspositionEntry;

/*
    Layers:
    In:
    6 curr piece boards
    6 opp piece boards
    3 current step
    1 banned pieces board
    1 phase (play or setup)
    1 trap squares

    Out:
    40 directional boards (subtract irrelevant squares)
    12 push pull boards
    1 pass bit
    1 setup squares (16 logits)
*/
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
    type Analyzer = GameAnalyzer<GameState, Action, Value, Engine, Mapper, TranspositionEntry>;

    fn analyzer(&self) -> Self::Analyzer {
        self.0.inner().analyzer()
    }
}

impl model::Info for Model {
    fn info(&self) -> &ModelInfo {
        self.0.inner().info()
    }
}
