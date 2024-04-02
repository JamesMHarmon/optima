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
    pub fn new(action_type: ActionType, coordinate: Coordinate) -> Self {
        (action_type, coordinate).into()
    }

    pub fn rotate(&self) -> Self {
        let shift = self.is_wall_placement();
        let coord = self.coord();
        let rotated_coord = coord.rotate(shift);
        let action_coord_idx = self.action_offset() + rotated_coord.index() as u8;

        Self { action_coord_idx }
    }

    pub fn vertical_symmetry(&self) -> Self {
        let shift = self.is_wall_placement();
        let coord = self.coord();
        let flipped_coord = coord.vertical_symmetry(shift);
        let action_coord_idx = self.action_offset() + flipped_coord.index() as u8;

        Self { action_coord_idx }
    }

    pub(crate) fn action_type(&self) -> ActionType {
        match self.action_coord_idx / BOARD_SIZE as u8 {
            0 => ActionType::PawnMove,
            1 => ActionType::HorizontalWall,
            2 => ActionType::VerticalWall,
            _ => panic!("action_coord_idx is not valid."),
        }
    }

    pub(crate) fn is_wall_placement(&self) -> bool {
        self.action_coord_idx >= BOARD_SIZE as u8
    }

    pub(crate) fn coord(&self) -> Coordinate {
        let coord_idx = self.action_coord_idx as usize % BOARD_SIZE;
        Coordinate::from_index(coord_idx)
    }

    pub(crate) fn action_offset(&self) -> u8 {
        (self.action_coord_idx / BOARD_SIZE as u8) * BOARD_SIZE as u8
    }
}

impl From<(ActionType, Coordinate)> for Action {
    fn from(value: (ActionType, Coordinate)) -> Self {
        let (action_type, coordinate) = value;

        let action_type_offset = match action_type {
            ActionType::PawnMove => 0,
            ActionType::HorizontalWall => BOARD_SIZE,
            ActionType::VerticalWall => BOARD_SIZE * 2,
        };

        Self {
            action_coord_idx: (action_type_offset + coordinate.index()) as u8,
        }
    }
}

impl FromStr for Action {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let coordinate = s[..2].parse()?;

        let action_type = match s.chars().nth(2) {
            None => Ok(ActionType::PawnMove),
            Some('h') => Ok(ActionType::HorizontalWall),
            Some('v') => Ok(ActionType::VerticalWall),
            Some(_) => Err(anyhow!("Invalid value")),
        };

        action_type.map(|action_type| Self::new(action_type, coordinate))
    }
}

impl Display for Action {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let action_type = match self.action_type() {
            ActionType::PawnMove => "",
            ActionType::HorizontalWall => "h",
            ActionType::VerticalWall => "v",
        };

        let coordinate = self.coord();

        write!(
            f,
            "{coordinate}{action_type}",
            coordinate = coordinate,
            action_type = action_type
        )
    }
}

impl Debug for Action {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self)
    }
}

#[derive(Clone, Copy)]
pub enum ActionType {
    PawnMove,
    HorizontalWall,
    VerticalWall,
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn all_coords_iter() -> impl Iterator<Item = Coordinate> {
        (0..81).map(Coordinate::from_index)
    }

    fn all_actions_iter() -> impl Iterator<Item = (Action, Coordinate)> {
        all_coords_iter().flat_map(|coord| {
            [
                Action::new(ActionType::PawnMove, coord),
                Action::new(ActionType::HorizontalWall, coord),
                Action::new(ActionType::VerticalWall, coord),
            ]
            .into_iter()
            .map(move |action| (action, coord))
        })
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
    fn test_action_to_string_for_all() {
        for (action, _) in all_actions_iter() {
            assert_eq!(action.to_string().parse::<Action>().unwrap(), action);
        }
    }

    #[test]
    fn action_coordinate_matches_coordinate_index_for_all() {
        for (action, coord) in all_actions_iter() {
            assert_eq!(Action::from(action).coord(), coord)
        }
    }
}
