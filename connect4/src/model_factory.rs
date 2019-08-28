use model::model_info::ModelInfo;

pub struct ModelFactory {}

impl ModelFactory {
    pub fn new() -> Self {
        Self {}
    }
}

impl model::model::ModelFactory for ModelFactory {
    type M = Model;

    fn create(&self, model_info: &ModelInfo, num_filters: usize, num_blocks: usize) -> Model {
        // @TODO: Replace with code to create the model.
        get_latest(model_info).expect("Failed to get latest model")        
    }

    fn get(&self, model_info: &ModelInfo) -> Model {
        Model::new(model_info.clone())
    }

    fn get_latest(&self, model_info: &ModelInfo) -> Model {
        get_latest(model_info).expect("Failed to get latest model")
    }
}


