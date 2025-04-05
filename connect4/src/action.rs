use serde::de::Error;
use serde::de::{Deserialize, Deserializer, Visitor};
use serde::ser::{Serialize, Serializer};
use std::fmt;
use std::fmt::{Display, Formatter};
use std::str::FromStr;

use anyhow::anyhow;

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum Action {
    DropPiece(u8),
}

impl Action {
    pub fn column(&self) -> u8 {
        match self {
            Action::DropPiece(column) => *column,
        }
    }
}

impl FromStr for Action {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let column_num = s.parse()?;

        if column_num > 7 {
            return Err(anyhow!("Column number must be between 1 and 7"));
        }

        Ok(Action::DropPiece(column_num))
    }
}

impl Display for Action {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let Action::DropPiece(column) = self;
        write!(f, "{}", column)
    }
}

impl Serialize for Action {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_u8(match self {
            Action::DropPiece(c) => *c,
        })
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
        Ok(Action::DropPiece(v))
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
