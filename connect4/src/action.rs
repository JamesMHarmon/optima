use serde::de::Error;
use std::fmt;
use serde::ser::{Serialize, Serializer};
use serde::de::{Deserialize, Deserializer, Visitor};

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum Action {
    DropPiece(u64)
}

impl Serialize for Action
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_u64(match self { Action::DropPiece(c) => *c })
    }
}

struct ActionVisitor {}

impl ActionVisitor {
    fn new() -> Self { Self {} }
}

impl<'de> Visitor<'de> for ActionVisitor
{
    type Value = Action;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("Expecting an integer from 1-7 that represents the column that a piece was dropped.")
    }

    fn visit_u64<E>(self, v: u64) -> Result<Self::Value, E>
    where
        E: Error,
    {
        Ok(Action::DropPiece(v as u64))
    }
}

impl<'de> Deserialize<'de> for Action
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_u64(ActionVisitor::new())
    }
}
