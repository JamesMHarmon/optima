use serde::de::Error;
use serde::de::{Deserialize, Deserializer, Visitor};
use serde::ser::{Serialize, Serializer};
use std::fmt;
use std::fmt::{Display, Formatter};
use std::str::FromStr;

use anyhow::anyhow;

#[derive(Debug, Clone, Eq, Hash, PartialEq)]
pub struct Action(u8);

impl Action {
    pub fn column(&self) -> usize {
        self.0 as usize
    }
}

impl From<usize> for Action {
    fn from(column: usize) -> Self {
        (column as u8).into()
    }
}

impl From<u8> for Action {
    fn from(column: u8) -> Self {
        Action(column)
    }
}

impl FromStr for Action {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let column_num: u8 = s.parse()?;

        if column_num > 7 {
            return Err(anyhow!("Column number must be between 1 and 7"));
        }

        Ok(column_num.into())
    }
}

impl Display for Action {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}", self.column())
    }
}

impl Serialize for Action {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_u8(self.column() as u8)
    }
}

struct ActionVisitor {}

impl ActionVisitor {
    fn new() -> Self {
        Self {}
    }
}

impl Visitor<'_> for ActionVisitor {
    type Value = Action;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str(
            "Expecting an integer from 1-7 that represents the column that a piece was dropped.",
        )
    }

    fn visit_u8<E>(self, v: u8) -> Result<Self::Value, E>
    where
        E: Error,
    {
        Ok(v.into())
    }
}

impl<'de> Deserialize<'de> for Action {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_u8(ActionVisitor::new())
    }
}
