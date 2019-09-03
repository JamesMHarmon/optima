use serde::de::Error;
use std::fmt;
use serde::ser::{Serialize,Serializer};
use serde::de::{Deserialize,Deserializer,Error as DeserializeError,Unexpected,Visitor};
use common::bits::single_bit_index;

#[derive(Clone,Debug,Eq,PartialEq)]
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
}

impl Serialize for Action
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let coordinate = match self {
            Action::MovePawn(coord) => format!("{}{}", coord.column, coord.row),
            Action::PlaceVerticalWall(coord) => format!("{}{}v", coord.column, coord.row),
            Action::PlaceHorizontalWall(coord) => format!("{}{}h", coord.column, coord.row)
        };

        serializer.serialize_str(&coordinate)
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
        let coordinate = Coordinate::new(chars[0],chars[1] as usize);

        match chars.get(3) {
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
        deserializer.deserialize_u64(ActionVisitor::new())
    }
}

#[derive(Clone,Debug,Eq,PartialEq)]
pub enum Action {
    MovePawn(Coordinate),
    PlaceHorizontalWall(Coordinate),
    PlaceVerticalWall(Coordinate)
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
}
