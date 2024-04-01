use crate::BOARD_SIZE;

use super::constants::{ASCII_LETTER_A, BOARD_HEIGHT, BOARD_WIDTH};
use anyhow::anyhow;
use common::bits::single_bit_index;
use serde::de::Error;
use serde::de::{Deserialize, Deserializer, Error as DeserializeError, Unexpected, Visitor};
use serde::ser::{Serialize, Serializer};
use std::fmt::{self, Debug, Display};
use std::str::FromStr;

#[derive(Clone, Copy, Eq, PartialEq)]
pub struct Coordinate {
    pub column: char,
    pub row: usize,
}

impl Coordinate {
    pub fn new(column: char, row: usize) -> Self {
        Self { column, row }
    }

    pub fn from_bit_board(board: u128) -> Self {
        let index = single_bit_index(board);
        let column = BOARD_WIDTH - (index % BOARD_WIDTH);
        let row = (index / BOARD_WIDTH) + 1;

        let column = (ASCII_LETTER_A + column as u8 - 1) as char;

        Coordinate::new(column, row)
    }

    pub fn as_bit_board(&self) -> u128 {
        let letter_pos: u8 = self.column as u8 - ASCII_LETTER_A + 1;
        let bit_in_column: u128 = 1 << (BOARD_WIDTH as u128 - letter_pos as u128);
        let row_shift = (self.row - 1) * BOARD_WIDTH;

        bit_in_column << row_shift
    }

    pub fn rotate(&self, shift: bool) -> Coordinate {
        Coordinate::new(
            if shift {
                Self::rotate_column_shift(self.column)
            } else {
                Self::rotate_column(self.column)
            },
            if shift {
                BOARD_HEIGHT
            } else {
                BOARD_HEIGHT + 1
            } - self.row,
        )
    }

    pub fn vertical_symmetry(&self, shift: bool) -> Coordinate {
        Coordinate::new(
            if shift {
                Self::rotate_column_shift(self.column)
            } else {
                Self::rotate_column(self.column)
            },
            self.row,
        )
    }

    pub fn index(&self) -> usize {
        let col_idx = (self.column as u8 - ASCII_LETTER_A) as usize;

        col_idx + ((BOARD_HEIGHT - self.row) * BOARD_WIDTH)
    }

    pub fn from_index(value: usize) -> Self {
        let col_idx = value % BOARD_WIDTH;
        let row_idx = BOARD_HEIGHT - 1 - value / BOARD_WIDTH;

        let col = (col_idx as u8 + ASCII_LETTER_A) as char;
        let row = row_idx + 1;

        Coordinate::new(col, row)
    }

    fn rotate_column(column: char) -> char {
        let col_num = column as u8 - ASCII_LETTER_A + 1;
        let rotated_col_num = BOARD_WIDTH as u8 - col_num + 1;
        (ASCII_LETTER_A + rotated_col_num - 1) as char
    }

    fn rotate_column_shift(column: char) -> char {
        let col_num = column as u8 - ASCII_LETTER_A + 1;
        let rotated_col_num = BOARD_WIDTH as u8 - col_num + 1;
        (ASCII_LETTER_A + rotated_col_num - 2) as char
    }
}

impl FromStr for Coordinate {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let chars: Vec<char> = s.chars().collect();
        let row = chars[0];
        let col = chars[1]
            .to_digit(10)
            .ok_or_else(|| anyhow!("Invalid value"))?;
        let coordinate = Coordinate::new(row, col as usize);
        Ok(coordinate)
    }
}

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
    compact: u8,
}

impl Action {
    pub fn rotate(&self) -> Self {
        ActionExpanded::from(*self).rotate().into()
    }

    pub fn vertical_symmetry(&self) -> Self {
        ActionExpanded::from(*self).vertical_symmetry().into()
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
            compact: (action_type_offset + coordinate.index()) as u8,
        }
    }
}

impl From<Action> for ActionExpanded {
    fn from(value: Action) -> Self {
        let action_type_offset = value.compact as usize / BOARD_SIZE;
        let index = value.compact as usize % BOARD_SIZE;

        match action_type_offset {
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

impl fmt::Display for Coordinate {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}{}", self.column, self.row)
    }
}

impl fmt::Debug for Coordinate {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self)
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn all_coords_iter() -> impl Iterator<Item = Coordinate> {
        let cols = || 'a'..='i';
        let rows = 1..=9;

        rows.into_iter()
            .rev()
            .flat_map(move |row| cols().map(move |col| Coordinate::new(col, row)))
    }

    #[test]
    fn test_as_bit_board_a1() {
        let bit = "a1".parse::<Coordinate>().unwrap().as_bit_board();
        let col = 1;
        let row = 1;

        assert_eq!(1 << ((9 - col) + (row - 1) * 9), bit);
    }

    #[test]
    fn test_as_bit_board_a9() {
        let bit = "a9".parse::<Coordinate>().unwrap().as_bit_board();
        let col = 1;
        let row = 9;

        assert_eq!(1 << ((9 - col) + (row - 1) * 9), bit);
    }

    #[test]
    fn test_as_bit_board_i1() {
        let bit = "i1".parse::<Coordinate>().unwrap().as_bit_board();
        let col = 9;
        let row = 1;

        assert_eq!(1 << ((9 - col) + (row - 1) * 9), bit);
    }

    #[test]
    fn test_as_bit_board_i9() {
        let bit = "i9".parse::<Coordinate>().unwrap().as_bit_board();
        let col = 9;
        let row = 9;

        assert_eq!(1 << ((9 - col) + (row - 1) * 9), bit);
    }

    #[test]
    fn test_as_bit_board_e5() {
        let bit = "e5".parse::<Coordinate>().unwrap().as_bit_board();
        let col = 5;
        let row = 5;

        assert_eq!(1 << ((9 - col) + (row - 1) * 9), bit);
    }

    #[test]
    fn test_from_bit_board_a1() {
        let col = 1;
        let row = 1;
        let bit_board = 1 << ((9 - col) + (row - 1) * 9);

        let coordinate = Coordinate::from_bit_board(bit_board);

        assert_eq!("a1".parse::<Coordinate>().unwrap(), coordinate);
    }

    #[test]
    fn test_from_bit_board_a9() {
        let col = 1;
        let row = 9;
        let bit_board = 1 << ((9 - col) + (row - 1) * 9);

        let coordinate = Coordinate::from_bit_board(bit_board);

        assert_eq!("a9".parse::<Coordinate>().unwrap(), coordinate);
    }

    #[test]
    fn test_from_bit_board_i1() {
        let col = 9;
        let row = 1;
        let bit_board = 1 << ((9 - col) + (row - 1) * 9);

        let coordinate = Coordinate::from_bit_board(bit_board);

        assert_eq!("i1".parse::<Coordinate>().unwrap(), coordinate);
    }

    #[test]
    fn test_from_bit_board_i9() {
        let col = 9;
        let row = 9;
        let bit_board = 1 << ((9 - col) + (row - 1) * 9);

        let coordinate = Coordinate::from_bit_board(bit_board);

        assert_eq!("i9".parse::<Coordinate>().unwrap(), coordinate);
    }

    #[test]
    fn test_from_bit_board_e5() {
        let col = 5;
        let row = 5;
        let bit_board = 1 << ((9 - col) + (row - 1) * 9);

        let coordinate = Coordinate::from_bit_board(bit_board);

        assert_eq!("e5".parse::<Coordinate>().unwrap(), coordinate);
    }

    #[test]
    fn test_to_coordinate_to_bit_board_from_bit_board_back_to_coordinate() {
        for orig_coordinate in all_coords_iter() {
            let bit_board = orig_coordinate.as_bit_board();
            let coordinate = Coordinate::from_bit_board(bit_board);

            assert_eq!(orig_coordinate, coordinate);
        }
    }

    #[test]
    fn test_rotate_coordinate_a1() {
        let coord = "a1".parse::<Coordinate>().unwrap();
        let expected = "i9".parse::<Coordinate>().unwrap();

        assert_eq!(coord.rotate(false), expected);
    }

    #[test]
    fn test_rotate_coordinate_i9() {
        let coord = "i9".parse::<Coordinate>().unwrap();
        let expected = "a1".parse::<Coordinate>().unwrap();

        assert_eq!(coord.rotate(false), expected);
    }

    #[test]
    fn test_rotate_coordinate_e5() {
        let coord = "e5".parse::<Coordinate>().unwrap();
        let expected = "e5".parse::<Coordinate>().unwrap();

        assert_eq!(coord.rotate(false), expected);
    }

    #[test]
    fn test_rotate_coordinate_d3() {
        let coord = "d3".parse::<Coordinate>().unwrap();
        let expected = "f7".parse::<Coordinate>().unwrap();

        assert_eq!(coord.rotate(false), expected);
    }

    #[test]
    fn test_rotate_coordinate_double_rotate() {
        let coord = "d3".parse::<Coordinate>().unwrap();

        assert_eq!(coord.rotate(false).rotate(false), coord);
    }

    #[test]
    fn test_rotate_coordinate_shift_a1() {
        let coord = "a1".parse::<Coordinate>().unwrap();
        let expected = "h8".parse::<Coordinate>().unwrap();

        assert_eq!(coord.rotate(true), expected);
    }

    #[test]
    fn test_rotate_coordinate_shift_h8() {
        let coord = "h8".parse::<Coordinate>().unwrap();
        let expected = "a1".parse::<Coordinate>().unwrap();

        assert_eq!(coord.rotate(true), expected);
    }

    #[test]
    fn test_rotate_coordinate_shift_e5() {
        let coord = "e5".parse::<Coordinate>().unwrap();
        let expected = "d4".parse::<Coordinate>().unwrap();

        assert_eq!(coord.rotate(true), expected);
    }

    #[test]
    fn test_rotate_coordinate_shift_d3() {
        let coord = "d3".parse::<Coordinate>().unwrap();
        let expected = "e6".parse::<Coordinate>().unwrap();

        assert_eq!(coord.rotate(true), expected);
    }

    #[test]
    fn test_rotate_coordinate_shift_double_rotate() {
        let coord = "d3".parse::<Coordinate>().unwrap();

        assert_eq!(coord.rotate(true).rotate(true), coord);
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
    fn test_coordinate_to_index_a9() {
        let coord = "a9".parse::<Coordinate>().unwrap();

        assert_eq!(coord.index(), 0);
    }

    #[test]
    fn test_coordinate_to_index_i9() {
        let coord = "i9".parse::<Coordinate>().unwrap();

        assert_eq!(coord.index(), 8);
    }

    #[test]
    fn test_coordinate_to_index_a1() {
        let coord = "a1".parse::<Coordinate>().unwrap();

        assert_eq!(coord.index(), 72);
    }

    #[test]
    fn test_coordinate_to_index_i1() {
        let coord = "i1".parse::<Coordinate>().unwrap();

        assert_eq!(coord.index(), 80);
    }

    #[test]
    fn test_coordinate_to_index_all() {
        for (i, coord) in all_coords_iter().enumerate() {
            assert_eq!(coord.index(), i);
        }
    }

    #[test]
    fn test_coordinate_to_from_index_all() {
        for coord in all_coords_iter() {
            assert_eq!(Coordinate::from_index(coord.index()), coord);
        }
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
