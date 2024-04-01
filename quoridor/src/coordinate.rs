use crate::BOARD_SIZE;

use super::constants::{ASCII_LETTER_A, BOARD_HEIGHT, BOARD_WIDTH};
use anyhow::anyhow;
use common::bits::single_bit_index;
use std::fmt::{self};
use std::str::FromStr;

#[derive(Clone, Copy, Eq, PartialEq)]
pub struct Coordinate {
    /// The index on the board of the coordinate.
    ///
    /// Represented starting in the upper left corner of a9 with an index 0 and working right to i9 with an index 8, ending at i1 with an index of 80.
    pub coord_index: u8,
}

impl Coordinate {
    pub fn new(column: char, row: usize) -> Self {
        let col_idx = (column as u8 - ASCII_LETTER_A) as usize;
        let coord_index = col_idx + ((BOARD_HEIGHT - row) * BOARD_WIDTH);
        let coord_index = coord_index as u8;

        Self { coord_index }
    }

    pub fn from_bit_board(board: u128) -> Self {
        let coord_index = (BOARD_SIZE - 1) - single_bit_index(board);
        let coord_index = coord_index as u8;
        Coordinate { coord_index }
    }

    pub fn as_bit_board(&self) -> u128 {
        1 << ((BOARD_SIZE - 1) - self.coord_index as usize)
    }

    pub fn rotate(&self, shift: bool) -> Self {
        let mut coord_index = BOARD_SIZE as u8 - 1 - self.coord_index;

        if shift {
            // Shift the board down.
            coord_index += BOARD_WIDTH as u8;
            // Shift the board left.
            coord_index -= 1
        }

        Self { coord_index }
    }

    pub fn vertical_symmetry(&self, shift: bool) -> Self {
        let row = self.coord_index as usize / BOARD_WIDTH;
        let column = self.coord_index as usize % BOARD_WIDTH;
        let mut new_col = (BOARD_WIDTH - 1) - column;

        if shift {
            // Shift the board left.
            new_col -= 1;
        }

        let coord_index = row * BOARD_WIDTH + new_col;
        let coord_index = coord_index as u8;

        Self { coord_index }
    }

    pub fn index(&self) -> usize {
        self.coord_index as usize
    }

    pub fn from_index(value: usize) -> Self {
        assert!(
            value < BOARD_SIZE,
            "Coordinate value must be less than {}",
            BOARD_SIZE
        );

        Self {
            coord_index: value as u8,
        }
    }

    pub fn col(&self) -> char {
        let col_idx = self.coord_index % BOARD_WIDTH as u8;

        (col_idx + ASCII_LETTER_A) as char
    }

    pub fn row(&self) -> usize {
        let row_idx = BOARD_HEIGHT as u8 - self.coord_index / BOARD_WIDTH as u8;
        let row = row_idx;
        row as usize
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

impl fmt::Display for Coordinate {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}{}", self.col(), self.row())
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

    fn all_coords_iter() -> impl Iterator<Item = Coordinate> {
        (0..81).map(Coordinate::from_index)
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
}
