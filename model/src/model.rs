use std::path::Path;

use anyhow::Result;
use serde::{Deserialize, Serialize};

use super::analytics::GameAnalyzer;
use super::model_info::ModelInfo;

pub trait Analyzer {
    type State;
    type Action;
    type Predictions;
    type Analyzer: GameAnalyzer<
        Action = Self::Action,
        State = Self::State,
        Predictions = Self::Predictions,
    >;

    fn analyzer(&self) -> Self::Analyzer;
}

pub trait Info {
    fn info(&self) -> &ModelInfo;
}

pub trait Latest {
    type MR;

    fn latest(&self) -> Result<Self::MR>;
}

pub trait Load {
    type MR;
    type M;

    fn load(&self, model_ref: &Self::MR) -> Result<Self::M>;
}

pub trait Move {
    type MR;

    fn move_to(&self, model: &Self::MR, path: &Path) -> Result<Self::MR>;

    fn copy_to(&self, model: &Self::MR, path: &Path) -> Result<Self::MR>;
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ModelOptions {
    pub number_of_filters: usize,
    pub number_of_residual_blocks: usize,
}

#[derive(Clone)]
pub struct TrainOptions {
    pub train_ratio: f32,
    pub train_batch_size: usize,
    pub epochs: usize,
    pub learning_rate: f32,
    pub max_grad_norm: f32,
    pub policy_loss_weight: f32,
    pub value_loss_weight: f32,
    pub moves_left_loss_weight: f32,
}
