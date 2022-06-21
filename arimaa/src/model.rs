use std::fs::File;
use std::path::Path;
use std::path::PathBuf;
use tar::Archive;

use crate::PlaceTranspositionEntry;
use crate::PlayTranspositionEntry;

use super::engine::Engine;
use super::game_state::GameState;
use super::place_mappings::Mapper as PlaceMapper;
use super::place_model::ModelFactory as PlaceModelFactory;
use super::play_mappings::Mapper as PlayMapper;
use super::play_model::ModelFactory as PlayModelFactory;
use super::value::Value;

use arimaa_engine::Action;
use futures::future::Either;
use model::{GameStateAnalysis, Latest, Load, ModelInfo};
use tensorflow_model::{unarchive, Archive as ArchiveModel, ArchiveAnalyzer};
use tensorflow_model::{GameAnalyzer, TensorflowModel, UnwrappedReceiver};

use anyhow::{Context, Result};
use tempfile::{tempdir, TempDir};

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
        let path = self.model_dir.join("state.json");
        let file = File::open(&path)
            .with_context(|| format!("Failed to find or load model latest file at: {:?}", path))?;
        let info: serde_json::Value = serde_json::from_reader(file)?;

        let latest_model_path = self.model_dir.join(
            info["latest"]
                .as_str()
                .expect("Latest property not a string."),
        );

        Ok(ModelRef(latest_model_path))
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

// #[allow(clippy::unnecessary_filter_map)]
// impl model::model::Model for Model {
//     type State = GameState;
//     type Action = Action;
//     type Value = Value;
//     type Analyzer = Analyzer;

//     fn get_model_info(&self) -> &ModelInfo {
//         self.play_model.get_model_info()
//     }

//     fn train<I>(
//         &self,
//         target_model_info: &ModelInfo,
//         sample_metrics: I,
//         options: &TrainOptions,
//     ) -> Result<()>
//     where
//         I: Iterator<Item = PositionMetrics<Self::State, Self::Action, Self::Value>>,
//     {
//         let mut place_samples = Vec::new();

//         let play_sample_iter = sample_metrics.filter_map(|sample_metric| {
//             if sample_metric.game_state.is_play_phase() {
//                 Some(sample_metric)
//             } else {
//                 place_samples.push(sample_metric);
//                 None
//             }
//         });

//         self.play_model
//             .train(target_model_info, play_sample_iter, options)?;

//         let place_model_info = map_to_place_model_info(target_model_info);

//         let max_grad_norm = std::env::var("PLACE_MAX_GRAD_NORM")
//             .map(|v| {
//                 v.parse::<f32>()
//                     .expect("PLACE_MAX_GRAD_NORM must be a valid number")
//             })
//             .unwrap_or(options.max_grad_norm);

//         let place_options = TrainOptions {
//             max_grad_norm,
//             ..(*options)
//         };
//         self.place_model
//             .train(&place_model_info, place_samples.into_iter(), &place_options)
//     }
// }

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
    let file = File::open(path)?;
    let mut archive = Archive::new(file);

    let tempdir = tempdir()?;

    archive.unpack(tempdir.path())?;

    Ok(tempdir)
}
