use super::{Coordinate, BOARD_SIZE};

use anyhow::anyhow;
use serde::de::Error;
use serde::de::{Deserialize, Deserializer, Error as DeserializeError, Unexpected, Visitor};
use serde::ser::{Serialize, Serializer};
use std::fmt::{self, Debug, Display};
use std::str::FromStr;

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

#[derive(Clone, Copy, Eq, PartialEq)]
pub struct Action {
    action_coord_idx: u8,
}

impl Action {
    pub fn rotate(&self) -> Self {
        ActionExpanded::from(*self).rotate().into()
    }

    pub fn vertical_symmetry(&self) -> Self {
        ActionExpanded::from(*self).vertical_symmetry().into()
    }

    pub(crate) fn coord(&self) -> usize {
        self.action_coord_idx as usize % BOARD_SIZE
    }
}

impl From<ActionExpanded> for Action {
    fn from(value: ActionExpanded) -> Self {
        let action_type_offset = match value {
            ActionExpanded::MovePawn(_) => 0,
            ActionExpanded::PlaceHorizontalWall(_) => BOARD_SIZE,
            ActionExpanded::PlaceVerticalWall(_) => BOARD_SIZE * 2,
        };

        let coordinate = match value {
            ActionExpanded::MovePawn(coordinate) => coordinate,
            ActionExpanded::PlaceHorizontalWall(coordinate) => coordinate,
            ActionExpanded::PlaceVerticalWall(coordinate) => coordinate,
        };

        Self {
            action_coord_idx: (action_type_offset + coordinate.index()) as u8,
        }
    }
}

impl From<Action> for ActionExpanded {
    fn from(value: Action) -> Self {
        let action_type = value.action_coord_idx as usize / BOARD_SIZE;
        let index = value.action_coord_idx as usize % BOARD_SIZE;

        match action_type {
            0 => ActionExpanded::MovePawn(Coordinate::from_index(index)),
            1 => ActionExpanded::PlaceHorizontalWall(Coordinate::from_index(index)),
            2 => ActionExpanded::PlaceVerticalWall(Coordinate::from_index(index)),
            _ => panic!("Invalid action type offset"),
        }
    }
}

impl FromStr for Action {
    type Err = <ActionExpanded as FromStr>::Err;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        s.parse::<ActionExpanded>().map(|a| a.into())
    }
}

impl Display for Action {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        Display::fmt(&ActionExpanded::from(*self), f)
    }
}

impl Debug for Action {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        Debug::fmt(&ActionExpanded::from(*self), f)
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
pub enum ActionExpanded {
    MovePawn(Coordinate),
    PlaceHorizontalWall(Coordinate),
    PlaceVerticalWall(Coordinate),
}

impl ActionExpanded {
    pub fn rotate(&self) -> Self {
        match self {
            Self::MovePawn(coordinate) => Self::MovePawn(coordinate.rotate(false)),
            Self::PlaceHorizontalWall(coordinate) => {
                Self::PlaceHorizontalWall(coordinate.rotate(true))
            }
            Self::PlaceVerticalWall(coordinate) => Self::PlaceVerticalWall(coordinate.rotate(true)),
        }
    }

    pub fn vertical_symmetry(&self) -> Self {
        match self {
            Self::MovePawn(coordinate) => Self::MovePawn(coordinate.vertical_symmetry(false)),
            Self::PlaceHorizontalWall(coordinate) => {
                Self::PlaceHorizontalWall(coordinate.vertical_symmetry(true))
            }
            Self::PlaceVerticalWall(coordinate) => {
                Self::PlaceVerticalWall(coordinate.vertical_symmetry(true))
            }
        }
    }
}

impl fmt::Display for ActionExpanded {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let (coordinate, action_type) = match self {
            Self::MovePawn(coordinate) => (coordinate, ""),
            Self::PlaceHorizontalWall(coordinate) => (coordinate, "h"),
            Self::PlaceVerticalWall(coordinate) => (coordinate, "v"),
        };

        write!(
            f,
            "{coordinate}{action_type}",
            coordinate = coordinate,
            action_type = action_type
        )
    }
}

impl fmt::Debug for ActionExpanded {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self)
    }
}

impl FromStr for ActionExpanded {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let coordinate = s[..2].parse()?;

        match s.chars().nth(2) {
            None => Ok(Self::MovePawn(coordinate)),
            Some('v') => Ok(Self::PlaceVerticalWall(coordinate)),
            Some('h') => Ok(Self::PlaceHorizontalWall(coordinate)),
            Some(_) => Err(anyhow!("Invalid value")),
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn all_coords_iter() -> impl Iterator<Item = Coordinate> {
        (0..81).map(Coordinate::from_index)
    }

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

    #[test]
    fn test_action_from_to_for_all() {
        let into_action_and_back =
            |action: ActionExpanded| ActionExpanded::from(Action::from(action));

        for coord in all_coords_iter() {
            let actions = [
                ActionExpanded::MovePawn(coord),
                ActionExpanded::PlaceHorizontalWall(coord),
                ActionExpanded::PlaceVerticalWall(coord),
            ];

            for action in actions.into_iter() {
                assert_eq!(action, into_action_and_back(action));
            }
        }
    }

    #[test]
    fn test_action_to_string_for_all() {
        for coord in all_coords_iter() {
            let actions = [
                ActionExpanded::MovePawn(coord),
                ActionExpanded::PlaceHorizontalWall(coord),
                ActionExpanded::PlaceVerticalWall(coord),
            ];

            for action in actions {
                assert_eq!(action.to_string(), Action::from(action).to_string());
                assert_eq!(
                    action.to_string().parse::<ActionExpanded>().unwrap(),
                    ActionExpanded::from(
                        Action::from(action).to_string().parse::<Action>().unwrap()
                    )
                );
            }
        }
    }
}
