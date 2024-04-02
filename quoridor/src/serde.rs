use serde::de::Error;
use serde::de::{Deserialize, Deserializer, Error as DeserializeError, Unexpected, Visitor};
use serde::ser::{Serialize, Serializer};
use std::fmt::{self};

use super::Action;

impl Serialize for Action {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&format!("{}", self))
    }
}

struct ActionVisitor {}

impl ActionVisitor {
    fn new() -> Self {
        Self {}
    }
}

impl<'de> Visitor<'de> for ActionVisitor {
    type Value = Action;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("Expecting a string with a letter representing the column then a number representing the row. Optionally followed by a 'v' or 'h' for a wall.")
    }

    fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
    where
        E: Error,
    {
        v.parse::<Action>()
            .map_err(|_| DeserializeError::invalid_value(Unexpected::Str(v), &self))
    }
}

impl<'de> Deserialize<'de> for Action {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_str(ActionVisitor::new())
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn test_action_pawn_move_ser_json() {
        let action = "a1".parse::<Action>().unwrap();
        let serialized_action_as_json = json!(action);

        assert_eq!(serialized_action_as_json, "a1");
    }

    #[test]
    fn test_action_vertical_wall_ser_json() {
        let action = "a1v".parse::<Action>().unwrap();
        let serialized_action_as_json = json!(action);

        assert_eq!(serialized_action_as_json, "a1v");
    }

    #[test]
    fn test_action_horizontal_wall_ser_json() {
        let action = "a1h".parse::<Action>().unwrap();
        let serialized_action_as_json = json!(action);

        assert_eq!(serialized_action_as_json, "a1h");
    }

    #[test]
    fn test_action_deser_pawn_move() {
        let json = "\"i9\"";

        assert_eq!(
            serde_json::from_str::<Action>(json).unwrap(),
            "i9".parse::<Action>().unwrap(),
        );
    }

    #[test]
    fn test_action_deser_horizontal_wall() {
        let json = "\"b6h\"";

        assert_eq!(
            serde_json::from_str::<Action>(json).unwrap(),
            "b6h".parse::<Action>().unwrap(),
        );
    }

    #[test]
    fn test_action_deser_vertical_wall() {
        let json = "\"d1v\"";

        assert_eq!(
            serde_json::from_str::<Action>(json).unwrap(),
            "d1v".parse::<Action>().unwrap(),
        );
    }
}
