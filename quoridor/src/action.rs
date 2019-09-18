use serde::de::Error;
use std::fmt;
use serde::ser::{Serialize,Serializer};
use serde::de::{Deserialize,Deserializer,Error as DeserializeError,Unexpected,Visitor};
use common::bits::single_bit_index;

#[derive(Clone,Eq,PartialEq)]
pub struct Coordinate {
    pub column: char,
    pub row: usize
}

impl Coordinate {
    pub fn new(column: char, row: usize) -> Self {
        Self { column, row }
    }

    pub fn from_bit_board(board: u128) -> Self {
        let index = single_bit_index(board);
        let column = 9 - (index % 9);
        let row = ((index / 9) as usize) + 1;

        let column = match column {
            1 => 'a',
            2 => 'b',
            3 => 'c',
            4 => 'd',
            5 => 'e',
            6 => 'f',
            7 => 'g',
            8 => 'h',
            _ => 'i',
        };

        Coordinate::new(column, row)
    }

    pub fn as_bit_board(&self) -> u128 {
        let col_bit = match self.column {
            'a' => 1 << 8,
            'b' => 1 << 7,
            'c' => 1 << 6,
            'd' => 1 << 5,
            'e' => 1 << 4,
            'f' => 1 << 3,
            'g' => 1 << 2,
            'h' => 1 << 1,
             _  => 1 << 0
        };

        col_bit << ((self.row - 1) * 9)
    }

    pub fn invert(&self, shift: bool) -> Coordinate {
        Coordinate::new(
            if shift { Self::invert_column_shift(self.column) } else { Self::invert_column(self.column) },
            if shift { 9 } else { 10 } - self.row
        )
    }

    fn invert_column(column: char) -> char {
        match column {
            'a' => 'i',
            'b' => 'h',
            'c' => 'g',
            'd' => 'f',
            'e' => 'e',
            'f' => 'd',
            'g' => 'c',
            'h' => 'b',
             _ => 'a'
        }
    }

    fn invert_column_shift(column: char) -> char {
        match column {
            'a' => 'h',
            'b' => 'g',
            'c' => 'f',
            'd' => 'e',
            'e' => 'd',
            'f' => 'c',
            'g' => 'b',
            'h' => 'a',
             _  => panic!("Can't shift from 'i'")
        }
    }
}

impl Serialize for Action
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&format!("{}",self))
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
        formatter.write_str("Expecting a string with a letter representing the column then a number representing the row. Optionally followed by a 'v' or 'h' for a wall.")
    }

    fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
    where
        E: Error,
    {
        let chars: Vec<char> = v.chars().collect();
        let row = chars[0];
        let col = chars[1].to_digit(10).ok_or_else(|| DeserializeError::invalid_value(Unexpected::Char(chars[1]), &self))?;
        let coordinate = Coordinate::new(row, col as usize);

        match chars.get(2) {
            None => Ok(Action::MovePawn(coordinate)),
            Some('v') => Ok(Action::PlaceVerticalWall(coordinate)),
            Some('h') => Ok(Action::PlaceHorizontalWall(coordinate)),
            Some(v) => Err(DeserializeError::invalid_value(Unexpected::Char(*v),&self))
        }
    }
}

impl<'de> Deserialize<'de> for Action
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_str(ActionVisitor::new())
    }
}

#[derive(Clone,Eq,PartialEq)]
pub enum Action {
    MovePawn(Coordinate),
    PlaceHorizontalWall(Coordinate),
    PlaceVerticalWall(Coordinate)
}

impl Action {
    pub fn invert(&self) -> Self {
        match self {
            Action::MovePawn(coordinate) => Action::MovePawn(coordinate.invert(false)),
            Action::PlaceHorizontalWall(coordinate) => Action::PlaceHorizontalWall(coordinate.invert(true)),
            Action::PlaceVerticalWall(coordinate) => Action::PlaceVerticalWall(coordinate.invert(true))
        }
    }
}

impl fmt::Display for Action {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let (action_type, coordinate) = match self {
            Action::MovePawn(coordinate) => ("", coordinate),
            Action::PlaceHorizontalWall(coordinate) => ("h", coordinate),
            Action::PlaceVerticalWall(coordinate) => ("v", coordinate)
        };

        write!(f, "{coordinate}{action_type}", action_type = action_type, coordinate = coordinate)
    }
}

impl fmt::Debug for Action {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self)
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
    use super::*;

    #[test]
    fn test_as_bit_board_a1() {
        let bit = Coordinate::new('a', 1).as_bit_board();
        let col = 1;
        let row = 1;

        assert_eq!(1 << ((9 - col) + (row - 1) * 9), bit);
    }

    #[test]
    fn test_as_bit_board_a9() {
        let bit = Coordinate::new('a', 9).as_bit_board();
        let col = 1;
        let row = 9;

        assert_eq!(1 << ((9 - col) + (row - 1) * 9), bit);
    }

    #[test]
    fn test_as_bit_board_i1() {
        let bit = Coordinate::new('i', 1).as_bit_board();
        let col = 9;
        let row = 1;

        assert_eq!(1 << ((9 - col) + (row - 1) * 9), bit);
    }

    #[test]
    fn test_as_bit_board_i9() {
        let bit = Coordinate::new('i', 9).as_bit_board();
        let col = 9;
        let row = 9;

        assert_eq!(1 << ((9 - col) + (row - 1) * 9), bit);
    }

    #[test]
    fn test_as_bit_board_e5() {
        let bit = Coordinate::new('e', 5).as_bit_board();
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

        assert_eq!(Coordinate::new('a', 1), coordinate);
    }

    #[test]
    fn test_from_bit_board_a9() {
        let col = 1;
        let row = 9;
        let bit_board = 1 << ((9 - col) + (row - 1) * 9);

        let coordinate = Coordinate::from_bit_board(bit_board);

        assert_eq!(Coordinate::new('a', 9), coordinate);
    }

    #[test]
    fn test_from_bit_board_i1() {
        let col = 9;
        let row = 1;
        let bit_board = 1 << ((9 - col) + (row - 1) * 9);

        let coordinate = Coordinate::from_bit_board(bit_board);

        assert_eq!(Coordinate::new('i', 1), coordinate);
    }

    #[test]
    fn test_from_bit_board_i9() {
        let col = 9;
        let row = 9;
        let bit_board = 1 << ((9 - col) + (row - 1) * 9);

        let coordinate = Coordinate::from_bit_board(bit_board);

        assert_eq!(Coordinate::new('i', 9), coordinate);
    }

    #[test]
    fn test_from_bit_board_e5() {
        let col = 5;
        let row = 5;
        let bit_board = 1 << ((9 - col) + (row - 1) * 9);

        let coordinate = Coordinate::from_bit_board(bit_board);

        assert_eq!(Coordinate::new('e', 5), coordinate);
    }

    #[test]
    fn test_to_coordinate_to_bit_board_from_bit_board_back_to_coordinate() {
        let cols = ['a','b','c','d','e','f','g','h','i'];
        let rows = [1,2,3,4,5,6,7,8,9];

        for (col, row) in cols.iter().zip(rows.iter()) {
            let orig_coordinate = Coordinate::new(*col, *row);
            let bit_board = orig_coordinate.as_bit_board();
            let coordinate = Coordinate::from_bit_board(bit_board);

            assert_eq!(orig_coordinate, coordinate);
        }
    }

    #[test]
    fn test_invert_coordinate_a1() {
        let coord = Coordinate::new('a', 1);
        let expected = Coordinate::new('i', 9);
        
        assert_eq!(coord.invert(false), expected);
    }

    #[test]
    fn test_invert_coordinate_i9() {
        let coord = Coordinate::new('i', 9);
        let expected = Coordinate::new('a', 1);
        
        assert_eq!(coord.invert(false), expected);
    }

    #[test]
    fn test_invert_coordinate_e5() {
        let coord = Coordinate::new('e', 5);
        let expected = Coordinate::new('e', 5);
        
        assert_eq!(coord.invert(false), expected);
    }

    #[test]
    fn test_invert_coordinate_d3() {
        let coord = Coordinate::new('d', 3);
        let expected = Coordinate::new('f', 7);
        
        assert_eq!(coord.invert(false), expected);
    }

    #[test]
    fn test_invert_coordinate_double_invert() {
        let coord = Coordinate::new('d', 3);
        
        assert_eq!(coord.invert(false).invert(false), coord);
    }

    #[test]
    fn test_invert_coordinate_shift_a1() {
        let coord = Coordinate::new('a', 1);
        let expected = Coordinate::new('h', 8);
        
        assert_eq!(coord.invert(true), expected);
    }

    #[test]
    fn test_invert_coordinate_shift_h8() {
        let coord = Coordinate::new('h', 8);
        let expected = Coordinate::new('a', 1);
        
        assert_eq!(coord.invert(true), expected);
    }

    #[test]
    fn test_invert_coordinate_shift_e5() {
        let coord = Coordinate::new('e', 5);
        let expected = Coordinate::new('d', 4);
        
        assert_eq!(coord.invert(true), expected);
    }

    #[test]
    fn test_invert_coordinate_shift_d3() {
        let coord = Coordinate::new('d', 3);
        let expected = Coordinate::new('e', 6);
        
        assert_eq!(coord.invert(true), expected);
    }

    #[test]
    fn test_invert_coordinate_shift_double_invert() {
        let coord = Coordinate::new('d', 3);
        
        assert_eq!(coord.invert(true).invert(true), coord);
    }
}

#[cfg(test)]
mod test {
    use serde_json::json;
    use super::*;

    #[test]
    fn test_action_pawn_move_ser_json() {
        let action = Action::MovePawn(Coordinate::new('a', 1));
        let serialized_action_as_json = json!(action);

        assert_eq!(
            serialized_action_as_json,
            "a1"
        );
    }

    #[test]
    fn test_action_vertical_wall_ser_json() {
        let action = Action::PlaceVerticalWall(Coordinate::new('a', 1));
        let serialized_action_as_json = json!(action);

        assert_eq!(
            serialized_action_as_json,
            "a1v"
        );
    }

    #[test]
    fn test_action_horizontal_wall_ser_json() {
        let action = Action::PlaceHorizontalWall(Coordinate::new('a', 1));
        let serialized_action_as_json = json!(action);

        assert_eq!(
            serialized_action_as_json,
            "a1h"
        );
    }

    #[test]
    fn test_action_deser_pawn_move() {
        let json = "\"i9\"";

        assert_eq!(
            serde_json::from_str::<Action>(&json).unwrap(),
            Action::MovePawn(Coordinate::new('i', 9)),
        );
    }

    #[test]
    fn test_action_deser_horizontal_wall() {
        let json = "\"b6h\"";

        assert_eq!(
            serde_json::from_str::<Action>(&json).unwrap(),
            Action::PlaceHorizontalWall(Coordinate::new('b', 6)),
        );
    }

    #[test]
    fn test_action_deser_vertical_wall() {
        let json = "\"d1v\"";

        assert_eq!(
            serde_json::from_str::<Action>(&json).unwrap(),
            Action::PlaceVerticalWall(Coordinate::new('d', 1)),
        );
    }
}
