use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

use anyhow::{anyhow, Context, Result};
use hocon::{Hocon, HoconLoader};

#[derive(Debug)]
pub struct ConfigLoader {
    hocon: Hocon,
    env: HashMap<String, String>,
    scope: String,
    path: PathBuf,
}

impl ConfigLoader {
    pub fn new(path: impl AsRef<Path>, scope: String) -> Result<Self> {
        let path = path.as_ref();
        assert!(path.is_file(), "The config file {:?} was not found", path);

        let env = std::env::vars().collect::<HashMap<_, _>>();

        let hocon = HoconLoader::new()
            .load_file(path)
            .with_context(|| format!("Failed to find or load config file at: {:?}", path))?
            .hocon()?;

        if !matches!(hocon, Hocon::Hash(_)) {
            return Err(anyhow!("Top level of config {:?} must be an object", path));
        }

        if let Hocon::Hash(ref hash) = hocon && hash.is_empty() {
            return Err(anyhow!("Configurations not found in file {:?}", path));
        }

        Ok(Self {
            hocon,
            env,
            scope,
            path: path.to_path_buf(),
        })
    }

    pub fn get(&self, name: &str) -> Option<Value> {
        if let Some(value) = self.env.get(name) {
            return Some(Value::String(value.clone()));
        }

        let scope = &self.hocon[self.scope.as_str()];
        if matches!(scope, Hocon::Hash(_)) {
            if let Some(value) = Self::map_hocon(scope, name) {
                return Some(value);
            }
        }

        Self::map_hocon(&self.hocon, name)
    }

    // Gets a path setting from the config and resolves the path to be relative to the config file.
    pub fn get_relative_path(&self, name: &str) -> Result<PathBuf> {
        self.get(name)
            .ok_or_else(|| anyhow!("Property {} not found in config.", name))?
            .as_string()
            .ok_or_else(|| anyhow!("Property {} not a valid string.", name))
            .and_then(|v| {
                self.path
                    .parent()
                    .ok_or_else(|| anyhow!("No parent directory for config {:?}", self.path))
                    .map(|p| p.join(v))
            })
    }

    pub fn load<T: Config>(&self) -> Result<T> {
        let res = T::load(self)?;
        Ok(res)
    }

    fn map_hocon(hocon: &Hocon, name: &str) -> Option<Value> {
        match &hocon[name] {
            Hocon::Real(f64) => Some(Value::Float(*f64 as f32)),
            Hocon::Integer(i64) => Some(Value::Integer(*i64 as usize)),
            Hocon::String(string) => Some(Value::String(string.clone())),
            Hocon::Boolean(bool) => Some(Value::Boolean(*bool)),
            _ => None,
        }
    }
}

#[derive(Debug)]
pub enum Value {
    String(String),
    Integer(usize),
    Float(f32),
    Boolean(bool),
}

impl Value {
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Boolean(val) => Some(*val),
            Value::String(val) => Hocon::String(val.clone()).as_bool(),
            _ => None,
        }
    }

    pub fn as_usize(&self) -> Option<usize> {
        match self {
            Value::Integer(val) => Some(*val),
            Value::String(val) => val.parse::<usize>().ok(),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Value::Float(val) => Some(*val),
            Value::Integer(val) => Some(*val as f32),
            Value::String(val) => val.parse::<f32>().ok(),
            _ => None,
        }
    }

    pub fn as_string(&self) -> Option<String> {
        match self {
            Value::String(val) => Some(val.clone()),
            Value::Boolean(true) => Some("true".to_string()),
            Value::Boolean(false) => Some("false".to_string()),
            Value::Float(val) => Some(val.to_string()),
            Value::Integer(val) => Some(val.to_string()),
        }
    }
}

pub trait Config {
    fn load(config: &ConfigLoader) -> Result<Self>
    where
        Self: Sized;
}
