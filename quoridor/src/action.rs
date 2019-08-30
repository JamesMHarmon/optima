#[derive(Clone,Debug)]
pub struct Coordinate {
    column: char,
    row: usize
}

impl Coordinate {
    pub fn new(column: char, row: usize) -> Self {
        Self { column, row }
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

#[derive(Clone,Debug)]
pub enum Action {
    MovePawn(Coordinate),
    PlaceHorizontalWall(Coordinate),
    PlaceVerticalWall(Coordinate)
}

#[derive(Debug)]
pub struct ValidActions {
    pub vertical_wall_placement: u128,
    pub horizontal_wall_placement: u128,
    pub pawn_moves: u128
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
}
