use super::model::{Model};
use super::super::model::{self, TrainOptions};

pub struct ModelFactory {}

impl ModelFactory {
    pub fn new() -> Self {
        Self {}
    }
}

impl model::ModelFactory for ModelFactory {
    type M = Model;

    fn create(&self, name: &str) -> Model {
        Model::new(name.to_owned())
    }

    fn train(&self, from_name: &str, target_name: &str, options: &TrainOptions) -> Model
    {
        // TODO: Implement training of the model.

        Model::new(target_name.to_owned())
    }
}