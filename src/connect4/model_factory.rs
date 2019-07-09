use std::env;

use pyo3::prelude::*;
use pyo3::types::{PyList,IntoPyDict};

use super::model::{Model};
use super::super::model;

pub struct ModelFactory {}

impl ModelFactory {
    pub fn new() -> Self {
        let result = Self::set_python_paths();

        if let Err(e) = result {
            let gil = Python::acquire_gil();
            let py = gil.python();
            e.print(py);
        }

        Self {}
    }

    fn set_python_paths() -> PyResult<()> {
        let gil = Python::acquire_gil();
        let py = gil.python();

        let current_dir_result = env::current_dir().expect("Could not get environment path");
        let env_path = current_dir_result.to_str().expect("Path not valid");
        println!("Env Path: {}", env_path);

        let sys = py.import("sys")?;
        let path = sys.get("path")?.downcast_ref::<PyList>()?;

        path.call_method("append", (env_path.to_owned(), ), None)?;
        path.call_method("append", ("/anaconda3/lib/python3.6".to_owned(), ), None)?;
        path.call_method("append", ("/anaconda3/lib/python3.6/lib-dynload".to_owned(), ), None)?;
        path.call_method("append", ("/anaconda3/lib/python3.6/site-packages", ), None)?;

        Ok(())
    }
}

impl model::ModelFactory for ModelFactory {
    type M = Model;

    fn create(&self, name: &str, num_filters: usize, num_blocks: usize) -> Model {
        let model = create_model(name, num_filters, num_blocks);

        model.map_err(|e| {
            let gil = Python::acquire_gil();
            let py = gil.python();
            e.print(py);
        }).expect("Failed to create model")
    }

    fn get_latest(&self, name: &str) -> Model {
        let model = get_latest(name);

        model.map_err(|e| {
            let gil = Python::acquire_gil();
            let py = gil.python();
            e.print(py);
        }).expect("Failed to get latest model")
    }
}

fn create_model(name: &str, num_filters: usize, num_blocks: usize) -> PyResult<Model> {
    let gil = Python::acquire_gil();
    let py = gil.python();
    let c4 = py.import("c4_model")?;

    let locals = [
        ("num_filters", num_filters),
        ("num_blocks", num_blocks)
    ].into_py_dict(py);

    locals.set_item("input_shape", (6, 7, 2))?;

    c4.call("create", (name,), Some(&locals))?;

    Ok(Model::new(name.to_owned()))
}

fn get_latest(name: &str) -> PyResult<Model> {
    let gil = Python::acquire_gil();
    let py = gil.python();
    let c4 = py.import("c4_model")?;
    let name: String = c4.call("get_latest", (name,), None)?.extract()?;

    Ok(Model::new(name))
}
