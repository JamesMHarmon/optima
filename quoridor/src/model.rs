use model::ModelInfo;
use tensorflow_model::Archive as ArchiveModel;
use tensorflow_model::{GameAnalyzer, TensorflowModel};

use super::{Action, Engine, GameState, Mapper, Predictions, TranspositionEntry};

pub type Analyzer =
    GameAnalyzer<GameState, Action, Predictions, Engine, Mapper, TranspositionEntry>;

pub struct Model(
    ArchiveModel<
        TensorflowModel<GameState, Action, Predictions, Engine, Mapper, TranspositionEntry>,
    >,
);

impl Model {
    pub fn new(
        model: ArchiveModel<
            TensorflowModel<GameState, Action, Predictions, Engine, Mapper, TranspositionEntry>,
        >,
    ) -> Self {
        Self(model)
    }
}

impl model::Analyzer for Model {
    type State = GameState;
    type Action = Action;
    type Predictions = Predictions;
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
