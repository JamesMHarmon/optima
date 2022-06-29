use anyhow::Result;
use serde::{Deserialize, Serialize};

use super::analytics::GameAnalyzer;
use super::model_info::ModelInfo;

pub trait Analyzer {
    type State;
    type Action;
    type Value;
    type Analyzer: GameAnalyzer<Action = Self::Action, State = Self::State, Value = Self::Value>;

    fn analyzer(&self) -> Self::Analyzer;
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

pub trait Info {
    fn info(&self) -> &ModelInfo;
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
