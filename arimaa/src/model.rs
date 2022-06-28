use std::io::Read;
use flate2::read::GzDecoder;
use tensorflow_model::latest;
use std::fs::File;
use std::path::Path;
use std::path::PathBuf;
use tar::Archive;

use super::engine::Engine;
use super::game_state::GameState;
use super::place_mappings::Mapper as PlaceMapper;
use super::place_model::ModelFactory as PlaceModelFactory;
use super::play_mappings::Mapper as PlayMapper;
use super::play_model::ModelFactory as PlayModelFactory;
use super::value::Value;
use super::{PlaceTranspositionEntry, PlayTranspositionEntry};

use anyhow::{Context, Result};
use arimaa_engine::Action;
use futures::future::Either;
use model::{GameStateAnalysis, Latest, Load, ModelInfo};
use tempfile::{tempdir, TempDir};
use tensorflow_model::{unarchive, Archive as ArchiveModel, ArchiveAnalyzer};
use tensorflow_model::{GameAnalyzer, TensorflowModel, UnwrappedReceiver};

#[derive(Debug, Eq, PartialEq)]
pub struct ModelRef(PathBuf);

#[derive(Default)]
pub struct ModelFactory {
    model_dir: PathBuf,
    play_model_factory: PlayModelFactory,
    place_model_factory: PlaceModelFactory,
}

impl ModelFactory {
    pub fn new(model_dir: PathBuf) -> Self {
        ModelFactory {
            model_dir,
            play_model_factory: PlayModelFactory::new(),
            place_model_factory: PlaceModelFactory::new(),
        }
    }
}

impl Latest for ModelFactory {
    type MR = ModelRef;

    fn latest(&self) -> Result<Self::MR> {
        latest(&self.model_dir).map(|p| ModelRef(p))
    }
}

impl Load for ModelFactory {
    type MR = ModelRef;
    type M = Model;

    fn load(&self, model_ref: &Self::MR) -> Result<Self::M> {
        let temp_dir =
            unsplit(&model_ref.0).with_context(|| format!("Failed to open {:?}", model_ref.0))?;
        let play_path = temp_dir.path().join("play.tar.gz");
        let place_path = temp_dir.path().join("place.tar.gz");

        let (play_model_temp_dir, play_options, play_model_info) = unarchive(play_path)?;
        let (place_model_temp_dir, place_options, place_model_info) = unarchive(place_path)?;

        let play_model = self.play_model_factory.load(
            play_model_temp_dir.path().to_path_buf(),
            play_options,
            play_model_info,
        )?;
        let place_model = self.place_model_factory.load(
            place_model_temp_dir.path().to_path_buf(),
            place_options,
            place_model_info,
        )?;

        let play_model = ArchiveModel::new(play_model, play_model_temp_dir);
        let place_model = ArchiveModel::new(place_model, place_model_temp_dir);

        Ok(Model {
            play_model,
            place_model,
        })
    }
}

pub struct Model {
    play_model: ArchiveModel<
        TensorflowModel<GameState, Action, Value, Engine, PlayMapper, PlayTranspositionEntry>,
    >,
    place_model: ArchiveModel<
        TensorflowModel<GameState, Action, Value, Engine, PlaceMapper, PlaceTranspositionEntry>,
    >,
}

impl model::Analyzer for Model {
    type State = GameState;
    type Action = Action;
    type Value = Value;
    type Analyzer = Analyzer;

    fn analyzer(&self) -> <Self as model::Analyzer>::Analyzer {
        Analyzer {
            play_analyzer: self.play_model.analyzer(),
            place_analyzer: self.place_model.analyzer(),
        }
    }
}

impl model::Info for Model {
    fn info(&self) -> &ModelInfo {
        self.play_model.inner().info()
    }
}

pub struct Analyzer {
    play_analyzer: ArchiveAnalyzer<
        GameAnalyzer<GameState, Action, Value, Engine, PlayMapper, PlayTranspositionEntry>,
    >,
    place_analyzer: ArchiveAnalyzer<
        GameAnalyzer<GameState, Action, Value, Engine, PlaceMapper, PlaceTranspositionEntry>,
    >,
}

#[allow(clippy::type_complexity)]
impl model::analytics::GameAnalyzer for Analyzer {
    type Future = Either<
        UnwrappedReceiver<GameStateAnalysis<Self::Action, Self::Value>>,
        UnwrappedReceiver<GameStateAnalysis<Self::Action, Self::Value>>,
    >;
    type Action = Action;
    type State = GameState;
    type Value = Value;

    fn get_state_analysis(&self, game_state: &Self::State) -> Self::Future {
        if game_state.is_play_phase() {
            Either::Left(self.play_analyzer.get_state_analysis(game_state))
        } else {
            Either::Right(self.place_analyzer.get_state_analysis(game_state))
        }
    }
}

fn unsplit(path: &Path) -> Result<TempDir> {
    let mut file: Box<dyn Read> = Box::new(File::open(path)?);

    if let Some(ext) = path.extension() {
        if ext.to_string_lossy() == "gz" {
            file = Box::new(GzDecoder::new(file));
        }
    }

    let mut archive = Archive::new(file);

    let tempdir = tempdir()?;

    archive.unpack(tempdir.path())?;

    Ok(tempdir)
}
