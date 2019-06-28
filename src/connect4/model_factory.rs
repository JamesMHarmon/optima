use std::env;

use pyo3::prelude::*;
use pyo3::types::{PyList};

use super::model::{Model};
use super::super::model::{self, TrainOptions};

pub struct ModelFactory {}

impl ModelFactory {
    pub fn new() -> Self {
        Self::set_python_paths();

        Self {}
    }

    fn set_python_paths() {
        let gil = Python::acquire_gil();
        let py = gil.python();

        let current_dir_result = env::current_dir().unwrap();
        let env_path = current_dir_result.to_str().ok_or("Path not valid").unwrap();
        println!("Env Path: {}", env_path);

        let sys = py.import("sys").unwrap();
        let path = sys.get("path").unwrap().downcast_ref::<PyList>().unwrap();

        path.call_method("append", (env_path.to_owned(), ), None).unwrap();
        path.call_method("append", ("/anaconda3/lib/python3.6".to_owned(), ), None).unwrap();
        path.call_method("append", ("/anaconda3/lib/python3.6/lib-dynload".to_owned(), ), None).unwrap();
        path.call_method("append", ("/anaconda3/lib/python3.6/site-packages", ), None).unwrap();
    }
}

impl model::ModelFactory for ModelFactory {
    type M = Model;

    fn create(&self, name: &str) -> Model {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let c4 = py.import("c4_model").unwrap();
        c4.call("create", (name,), None).unwrap();

        Model::new(name.to_owned())
    }

    fn get_latest(&self, name: &str) -> Model {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let c4 = py.import("c4_model").unwrap();
        let name: String = c4.call("get_latest", (name,), None).unwrap().extract().unwrap();

        Model::new(name)
    }

    fn train(&self, from_name: &str, target_name: &str, options: &TrainOptions) -> Model
    {
        // TODO: Implement training of the model.

        Model::new(target_name.to_owned())
    }
}
