use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, anyhow};
use hocon_rs::{Config as HoconConfig, ConfigOptions, Value as HoconValue};

#[derive(Debug)]
pub struct ConfigLoader {
    hocon: HoconValue,
    env: HashMap<String, String>,
    scope: String,
    path: PathBuf,
}

impl ConfigLoader {
    pub fn new(path: impl AsRef<Path>, scope: String) -> Result<Self> {
        let path = path.as_ref();
        assert!(path.is_file(), "The config file {:?} was not found", path);

        let env = std::env::vars().collect::<HashMap<_, _>>();

        let mut options = ConfigOptions::default();
        if let Some(parent) = path.parent().and_then(|p| p.to_str()) {
            options.classpath = vec![parent.to_string()].into();
        }

        let hocon = HoconConfig::parse_file::<HoconValue>(path, Some(options))
            .with_context(|| format!("Failed to find or load config file at: {:?}", path))?;

        let Some(hash) = hocon.as_object() else {
            return Err(anyhow!("Top level of config {:?} must be an object", path));
        };

        if hash.is_empty() {
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

        if let Some(scope_value) = self
            .hocon
            .as_object()
            .and_then(|hash| hash.get(self.scope.as_str()))
            .and_then(|scope_value| Self::map_hocon(scope_value, name))
        {
            return Some(scope_value);
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

    fn map_hocon(hocon: &HoconValue, name: &str) -> Option<Value> {
        let value = hocon.as_object().and_then(|hash| hash.get(name))?;

        match value {
            HoconValue::Number(number) => {
                if let Some(f64) = number.as_f64() {
                    return Some(Value::Float(f64 as f32));
                }

                if let Some(u64) = number.as_u64() {
                    return Some(Value::Integer(u64 as usize));
                }

                if let Some(i64) = number.as_i64() {
                    return usize::try_from(i64).ok().map(Value::Integer);
                }

                None
            }
            HoconValue::String(string) => Some(Value::String(string.clone())),
            HoconValue::Boolean(boolean) => Some(Value::Boolean(*boolean)),
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
            Value::String(val) => {
                if val.eq_ignore_ascii_case("true") {
                    Some(true)
                } else if val.eq_ignore_ascii_case("false") {
                    Some(false)
                } else {
                    None
                }
            }
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
