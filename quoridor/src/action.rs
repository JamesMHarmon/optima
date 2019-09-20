use super::constants::{BOARD_HEIGHT,BOARD_WIDTH,ASCII_LETTER_A};
use std::str::FromStr;
use serde::de::Error;
use std::fmt;
use serde::ser::{Serialize,Serializer};
use serde::de::{Deserialize,Deserializer,Error as DeserializeError,Unexpected,Visitor};
use common::bits::single_bit_index;
use failure::{format_err};

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
        let column = BOARD_WIDTH - (index % BOARD_WIDTH);
        let row = ((index / BOARD_WIDTH) as usize) + 1;

        let column = (ASCII_LETTER_A  + column as u8 - 1) as char;

        Coordinate::new(column, row)
    }

    pub fn as_bit_board(&self) -> u128 {
        let letter_pos: u8 = self.column as u8 - ASCII_LETTER_A + 1;
        let bit_in_column: u128 = 1 << (BOARD_WIDTH as u128 - letter_pos as u128);
        let row_shift = (self.row - 1) * BOARD_HEIGHT;

        bit_in_column << row_shift
    }

    pub fn invert(&self, shift: bool) -> Coordinate {
        Coordinate::new(
            if shift { Self::invert_column_shift(self.column) } else { Self::invert_column(self.column) },
            if shift { BOARD_HEIGHT } else { BOARD_HEIGHT + 1 } - self.row
        )
    }

    fn invert_column(column: char) -> char {
        let col_num = column as u8 - ASCII_LETTER_A + 1;
        let inverted_col_num = BOARD_WIDTH as u8 - col_num + 1;
        let a = (ASCII_LETTER_A + inverted_col_num - 1) as char;

        a
    }

    fn invert_column_shift(column: char) -> char {
        let col_num = column as u8 - ASCII_LETTER_A + 1;
        let inverted_col_num = BOARD_WIDTH as u8 - col_num + 1;
        let a = (ASCII_LETTER_A + inverted_col_num - 2) as char;

        a
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
        v.parse::<Action>().map_err(|_| DeserializeError::invalid_value(Unexpected::Str(v),&self))
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
        let (coordinate,action_type) = match self {
            Action::MovePawn(coordinate) => (coordinate, ""),
            Action::PlaceHorizontalWall(coordinate) => (coordinate, "h"),
            Action::PlaceVerticalWall(coordinate) => (coordinate, "v")
        };

        write!(f, "{coordinate}{action_type}", coordinate = coordinate, action_type = action_type)
    }
}

impl fmt::Debug for Action {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self)
    }
}

impl FromStr for Action {
    type Err = failure::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let chars: Vec<char> = s.chars().collect();
        let row = chars[0];
        // @TODO: Update to take multiple digits
        let col = chars[1].to_digit(10).ok_or_else(|| format_err!("Invalid value"))?;
        let coordinate = Coordinate::new(row, col as usize);

        match chars.get(2) {
            None => Ok(Action::MovePawn(coordinate)),
            Some('v') => Ok(Action::PlaceVerticalWall(coordinate)),
            Some('h') => Ok(Action::PlaceHorizontalWall(coordinate)),
            Some(_) => Err(format_err!("Invalid value"))
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
