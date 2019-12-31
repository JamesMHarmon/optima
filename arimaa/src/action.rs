use super::constants::{BOARD_HEIGHT,BOARD_WIDTH,ASCII_LETTER_A};
use std::str::FromStr;
use serde::de::Error;
use std::fmt;
use serde::ser::{Serialize,Serializer};
use serde::de::{Deserialize,Deserializer,Error as DeserializeError,Unexpected,Visitor};
use common::bits::single_bit_index;
use failure::{format_err};

#[derive(Hash, Eq, PartialEq, Clone, Copy, Ord, PartialOrd)]
pub struct Square(u8);

impl Square {
    pub fn new(column: char, row: usize) -> Self {
        let index = (column as u8 - ASCII_LETTER_A) + (BOARD_HEIGHT - row) as u8 * 8;
        Square(index)
    }

    pub fn from_index(index: u8) -> Self {
        Square(index)
    }

    pub fn from_bit_board(board: u64) -> Self {
        Square(single_bit_index(board as u128) as u8)
    }

    pub fn as_bit_board(&self) -> u64 {
        1 << self.0
    }

    pub fn get_index(&self) -> usize {
        self.0 as usize
    }

    pub fn invert(&self) -> Self {
        Square((BOARD_HEIGHT * BOARD_WIDTH) as u8 - 1 - self.0)
    }

    pub fn invert_horizontal(&self) -> Self {
        Square(self.0 ^ 7)
    }

    pub fn column_char(&self) -> char {
        let index = self.0 as usize;
        let column = index % BOARD_WIDTH;

        (ASCII_LETTER_A + column as u8) as char
    }

    pub fn row(&self) -> u8 {
        let index = self.0 as usize;
        let row = BOARD_HEIGHT - (index / BOARD_WIDTH) as usize;
        row as u8
    }
}

impl fmt::Display for Square {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let column = self.column_char();
        let row = self.row();

        write!(f, "{column}{row}", column = column, row = row)
    }
}

impl fmt::Debug for Square {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self)
    }
}


impl FromStr for Square {
    type Err = failure::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let chars: Vec<char> = s.chars().collect();

        if chars.len() == 2 {
            let column = chars[0];
            let row = chars[1];

            if let Ok(row) = row.to_string().parse() {
                let column_as_num = column as u8 - ASCII_LETTER_A + 1;
                if column_as_num >= 1 && column_as_num <= BOARD_WIDTH as u8 && row >= 1 && row <= BOARD_HEIGHT {
                    return Ok(Square::new(column, row));
                }
            }
        }

        Err(format_err!("Invalid value for square"))
    }
}

#[derive(Clone,Copy,Eq,Hash,PartialEq)]
pub enum Direction {
    Up,
    Right,
    Down,
    Left
}

impl Direction {
    pub fn invert(&self) -> Self {
        match self {
            Direction::Up => Direction::Down,
            Direction::Right => Direction::Left,
            Direction::Down => Direction::Up,
            Direction::Left => Direction::Right
        }
    }

    pub fn invert_horizontal(&self) -> Self {
        match self {
            Direction::Up => Direction::Up,
            Direction::Right => Direction::Left,
            Direction::Down => Direction::Down,
            Direction::Left => Direction::Right
        }
    }
}

impl fmt::Display for Direction {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let dir = match self {
            Direction::Up => 'n',
            Direction::Right => 'e',
            Direction::Down => 's',
            Direction::Left => 'w'
        };

        write!(f, "{}", dir)
    }
}

impl FromStr for Direction {
    type Err = failure::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let chars: Vec<char> = s.chars().collect();

        if chars.len() == 1 {
            if let Some(c) = chars.get(0) {
                let direction = match c {
                    'n' => Some(Direction::Up),
                    'e' => Some(Direction::Right),
                    's' => Some(Direction::Down),
                    'w' => Some(Direction::Left),
                    _ => None
                };

                if let Some(direction) = direction {
                    return Ok(direction);
                }
            }
        }

        Err(format_err!("Invalid value for direction"))
    }
}

#[derive(Hash, Eq, PartialEq, PartialOrd, Ord, Clone, Copy, Debug)]
pub enum Piece {
    Rabbit,
    Cat,
    Dog,
    Horse,
    Camel,
    Elephant
}

impl fmt::Display for Piece {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let piece = match self {
            Piece::Elephant => 'e',
            Piece::Camel => 'm',
            Piece::Horse => 'h',
            Piece::Dog => 'd',
            Piece::Cat => 'c',
            Piece::Rabbit => 'r'
        };

        write!(f, "{}", piece)
    }
}

impl FromStr for Piece {
    type Err = failure::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let chars: Vec<char> = s.chars().collect();

        if chars.len() == 1 {
            if let Some(c) = chars.get(0) {
                let piece = match c {
                    'E' | 'e' => Some(Piece::Elephant),
                    'M' | 'm' => Some(Piece::Camel),
                    'H' | 'h' => Some(Piece::Horse),
                    'D' | 'd' => Some(Piece::Dog),
                    'C' | 'c' => Some(Piece::Cat),
                    'R' | 'r' => Some(Piece::Rabbit),
                    _ => None
                };

                if let Some(piece) = piece {
                    return Ok(piece);
                }
            }
        }

        Err(format_err!("Invalid value for piece"))
    }
}

#[derive(Hash, Clone, Eq, PartialEq)]
pub enum Action {
    Place(Piece),
    Move(Square,Direction),
    Pass
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
        formatter.write_str("Expecting a string with a letter representing the column then a number representing the row.")
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

impl Action {
    pub fn invert(&self) -> Self {
        match self {
            Action::Move(square,direction) => Action::Move(square.invert(),direction.invert()),
            Action::Place(_) => panic!("Cannot invert placement"),
            Action::Pass => Action::Pass
        }
    }

    pub fn invert_horizontal(&self) -> Self {
        match self {
            Action::Move(square,direction) => Action::Move(square.invert_horizontal(), direction.invert_horizontal()),
            Action::Place(_) => panic!("Cannot invert placement"),
            Action::Pass => Action::Pass
        }
    }
}

impl fmt::Display for Action {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let action = match self {
            Action::Move(square,direction) => format!("{}{}", square, direction),
            Action::Pass => "p".to_string(),
            Action::Place(piece) => format!("{}", piece)
        };

        write!(f, "{}", action)
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

        if chars.len() == 1 {
            if let Some(c) = chars.get(0) {
                if *c == 'p' {
                    return Ok(Action::Pass);
                } else if let Ok(piece) = c.to_string().parse::<Piece>() {
                    return Ok(Action::Place(piece));
                }
            }
        } else if chars.len() == 3 {
            if let Ok(square) = s[..2].parse::<Square>() {
                if let Ok(dir) = s[2..].parse::<Direction>() {
                    return Ok(Action::Move(square, dir));
                }
            }
        }

        Err(format_err!("Invalid action"))
    }
}

pub fn map_bit_board_to_squares(board: u64) -> Vec<Square> {
    let mut board = board;
    let mut squares = Vec::with_capacity(board.count_ones() as usize);

    while board != 0 {
        let bit_idx = board.trailing_zeros();
        let square = Square::from_index(bit_idx as u8);
        squares.push(square);

        board = board ^ 1 << bit_idx;
    }

    squares
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cmp::Ordering;

    fn col_row_to_bit(col: usize, row: usize) -> u64 {
        1 << ((col - 1) + (8 - row) * 8)
    }

    #[test]
    fn test_as_bit_board_a1() {
        let bit = Square::new('a', 1).as_bit_board();
        let col = 1;
        let row = 1;

        assert_eq!(col_row_to_bit(col, row), bit);
    }

    #[test]
    fn test_as_bit_board_a8() {
        let bit = Square::new('a', 8).as_bit_board();
        let col = 1;
        let row = 8;

        assert_eq!(col_row_to_bit(col, row), bit);
    }

    #[test]
    fn test_as_bit_board_h1() {
        let bit = Square::new('h', 1).as_bit_board();
        let col = 8;
        let row = 1;

        assert_eq!(col_row_to_bit(col, row), bit);
    }
    #[test]
    fn test_as_bit_board_h8() {
        let bit = Square::new('h', 8).as_bit_board();
        let col = 8;
        let row = 8;

        assert_eq!(col_row_to_bit(col, row), bit);
    }

    #[test]
    fn test_as_bit_board_e5() {
        let bit = Square::new('e', 5).as_bit_board();
        let col = 5;
        let row = 5;

        assert_eq!(col_row_to_bit(col, row), bit);
    }

    #[test]
    fn test_from_bit_board_a1() {
        let col = 1;
        let row = 1;
        let bit_board = col_row_to_bit(col, row);

        let square = Square::from_bit_board(bit_board);

        assert_eq!(Square::new('a', 1), square);
    }

    #[test]
    fn test_from_bit_board_a8() {
        let col = 1;
        let row = 8;
        let bit_board = col_row_to_bit(col, row);

        let square = Square::from_bit_board(bit_board);

        assert_eq!(Square::new('a', 8), square);
    }

    #[test]
    fn test_from_bit_board_h1() {
        let col = 8;
        let row = 1;
        let bit_board = col_row_to_bit(col, row);

        let square = Square::from_bit_board(bit_board);

        assert_eq!(Square::new('h', 1), square);
    }

    #[test]
    fn test_from_bit_board_h8() {
        let col = 8;
        let row = 8;
        let bit_board = col_row_to_bit(col, row);

        let square = Square::from_bit_board(bit_board);

        assert_eq!(Square::new('h', 8), square);
    }

    #[test]
    fn test_from_bit_board_e5() {
        let col = 5;
        let row = 5;
        let bit_board = col_row_to_bit(col, row);

        let square = Square::from_bit_board(bit_board);

        assert_eq!(Square::new('e', 5), square);
    }

    #[test]
    fn test_to_square_to_bit_board_from_bit_board_back_to_square() {
        let cols = ['a','b','c','d','e','f','g','h'];
        let rows = [1,2,3,4,5,6,7,8];

        for (col, row) in cols.iter().zip(rows.iter()) {
            let orig_square = Square::new(*col, *row);
            let bit_board = orig_square.as_bit_board();
            let square = Square::from_bit_board(bit_board);

            assert_eq!(orig_square, square);
        }
    }

    #[test]
    fn test_invert_square_a1() {
        let square = Square::new('a', 1);
        let expected = Square::new('h', 8);

        assert_eq!(square.invert(), expected);
    }

    #[test]
    fn test_invert_square_h8() {
        let square = Square::new('h', 8);
        let expected = Square::new('a', 1);

        assert_eq!(square.invert(), expected);
    }

    #[test]
    fn test_invert_square_e5() {
        let square = Square::new('e', 5);
        let expected = Square::new('d', 4);

        assert_eq!(square.invert(), expected);
    }

    #[test]
    fn test_invert_square_d3() {
        let square = Square::new('d', 3);
        let expected = Square::new('e', 6);

        assert_eq!(square.invert(), expected);
    }

    #[test]
    fn test_invert_square_horizontal_a1() {
        let square = Square::new('a', 1);
        let expected = Square::new('h', 1);

        assert_eq!(square.invert_horizontal(), expected);
    }

    #[test]
    fn test_invert_square_horizontal_h8() {
        let square = Square::new('h', 8);
        let expected = Square::new('a', 8);

        assert_eq!(square.invert_horizontal(), expected);
    }

    #[test]
    fn test_invert_square_horizontal_e5() {
        let square = Square::new('e', 5);
        let expected = Square::new('d', 5);

        assert_eq!(square.invert_horizontal(), expected);
    }

    #[test]
    fn test_invert_square_horizontal_d3() {
        let square = Square::new('d', 3);
        let expected = Square::new('e', 3);

        assert_eq!(square.invert_horizontal(), expected);
    }

    #[test]
    fn test_invert_square_horizontal_double_invert() {
        let square = Square::new('d', 3);

        assert_eq!(square.invert_horizontal().invert_horizontal(), square);
    }

    #[test]
    fn test_piece_precedence_rabbit() {
        assert_eq!(Piece::Rabbit.cmp(&Piece::Cat), Ordering::Less);
        assert_eq!(Piece::Rabbit.cmp(&Piece::Rabbit), Ordering::Equal);
        assert_eq!(Piece::Cat.cmp(&Piece::Rabbit), Ordering::Greater);
    }

    #[test]
    fn test_piece_precedence_cat() {
        assert_eq!(Piece::Cat.cmp(&Piece::Dog), Ordering::Less);
        assert_eq!(Piece::Cat.cmp(&Piece::Cat), Ordering::Equal);
        assert_eq!(Piece::Dog.cmp(&Piece::Cat), Ordering::Greater);
    }

    #[test]
    fn test_piece_precedence_dog() {
        assert_eq!(Piece::Dog.cmp(&Piece::Horse), Ordering::Less);
        assert_eq!(Piece::Dog.cmp(&Piece::Dog), Ordering::Equal);
        assert_eq!(Piece::Horse.cmp(&Piece::Dog), Ordering::Greater);
    }

    #[test]
    fn test_piece_precedence_horse() {
        assert_eq!(Piece::Horse.cmp(&Piece::Camel), Ordering::Less);
        assert_eq!(Piece::Horse.cmp(&Piece::Horse), Ordering::Equal);
        assert_eq!(Piece::Camel.cmp(&Piece::Horse), Ordering::Greater);
    }

    #[test]
    fn test_piece_precedence_camel() {
        assert_eq!(Piece::Camel.cmp(&Piece::Elephant), Ordering::Less);
        assert_eq!(Piece::Camel.cmp(&Piece::Camel), Ordering::Equal);
        assert_eq!(Piece::Elephant.cmp(&Piece::Camel), Ordering::Greater);
    }

    #[test]
    fn test_piece_precedence_elephant() {
        assert_eq!(Piece::Elephant.cmp(&Piece::Elephant), Ordering::Equal);
        assert_eq!(Piece::Elephant.cmp(&Piece::Rabbit), Ordering::Greater);
    }
}

#[cfg(test)]
mod test {
    use serde_json::json;
    use super::*;

    #[test]
    fn test_action_move_ser_json() {
        let action = Action::Move(Square::new('a', 1),Direction::Up);
        let serialized_action_as_json = json!(action);

        assert_eq!(
            serialized_action_as_json,
            "a1n"
        );
    }

    #[test]
    fn test_action_move_ser_json_2() {
        let action = Action::Move(Square::new('d', 4),Direction::Right);
        let serialized_action_as_json = json!(action);

        assert_eq!(
            serialized_action_as_json,
            "d4e"
        );
    }

    #[test]
    fn test_action_place_elephant_ser_json() {
        let action = Action::Place(Piece::Elephant);
        let serialized_action_as_json = json!(action);

        assert_eq!(
            serialized_action_as_json,
            "e"
        );
    }

    #[test]
    fn test_action_place_camel_ser_json() {
        let action = Action::Place(Piece::Camel);
        let serialized_action_as_json = json!(action);

        assert_eq!(
            serialized_action_as_json,
            "m"
        );
    }

    #[test]
    fn test_action_pass_ser_json() {
        let action = Action::Pass;
        let serialized_action_as_json = json!(action);

        assert_eq!(
            serialized_action_as_json,
            "p"
        );
    }

    #[test]
    fn test_action_deser_move() {
        let json = "\"b2w\"";

        assert_eq!(
            serde_json::from_str::<Action>(&json).unwrap(),
            Action::Move(Square::new('b', 2), Direction::Left),
        );
    }

    #[test]
    fn test_action_deser_place() {
        let json = "\"d\"";

        assert_eq!(
            serde_json::from_str::<Action>(&json).unwrap(),
            Action::Place(Piece::Dog),
        );
    }

    #[test]
    fn test_action_deser_pass() {
        let json = "\"p\"";

        assert_eq!(
            serde_json::from_str::<Action>(&json).unwrap(),
            Action::Pass,
        );
    }
}
