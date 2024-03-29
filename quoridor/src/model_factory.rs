use anyhow::Result;
use common::get_env_usize;
use std::fs;
use std::path::{Path, PathBuf};

use super::engine::Engine;
use super::mappings::Mapper;
use super::Model;
use model::{Latest, Load, Move};
use tensorflow_model::TensorflowModel;
use tensorflow_model::{latest, unarchive, Archive as ArchiveModel};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelRef(PathBuf);

impl ModelRef {
    pub fn new(path: PathBuf) -> Self {
        Self(path)
    }
}

#[derive(Default)]
pub struct ModelFactory {
    model_dir: PathBuf,
}

impl ModelFactory {
    pub fn new(model_dir: PathBuf) -> Self {
        Self { model_dir }
    }
}

impl Latest for ModelFactory {
    type MR = ModelRef;

    fn latest(&self) -> Result<Self::MR> {
        latest(&self.model_dir).map(ModelRef)
    }
}

impl Load for ModelFactory {
    type MR = ModelRef;
    type M = Model;

    fn load(&self, model_ref: &Self::MR) -> Result<Self::M> {
        let table_size = get_env_usize("TABLE_SIZE").unwrap_or(0);

        let (model_temp_dir, model_options, model_info) = unarchive(&model_ref.0)?;

        let tensorflow_model = TensorflowModel::load(
            model_temp_dir.path().to_path_buf(),
            model_options,
            model_info,
            Engine::new(),
            Mapper::new(),
            table_size,
        )?;

        let archive_model = ArchiveModel::new(tensorflow_model, model_temp_dir);

        Ok(Model::new(archive_model))
    }
}

impl Move for ModelFactory {
    type MR = ModelRef;

    fn move_to(&self, model: &Self::MR, path: &Path) -> Result<Self::MR> {
        let file_name = model.0.file_name().expect("Model has no file name");
        let file_path = path.join(file_name);
        fs::rename(&model.0, &file_path)?;

        Ok(ModelRef(file_path))
    }

    fn copy_to(&self, model: &Self::MR, path: &Path) -> Result<Self::MR> {
        let file_name = model.0.file_name().expect("Model has no file name");
        let file_path = path.join(file_name);
        fs::copy(&model.0, &file_path)?;

        Ok(ModelRef(file_path))
    }
}
