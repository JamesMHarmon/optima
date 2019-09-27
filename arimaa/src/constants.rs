pub const MAX_NUMBER_OF_MOVES: usize = 128;
pub const BOARD_WIDTH: usize = 8;
pub const BOARD_HEIGHT: usize = 8;


pub const INPUT_H: usize = BOARD_HEIGHT;
pub const INPUT_W: usize = BOARD_WIDTH;
pub const INPUT_C: usize = 6;
pub const PAWN_BOARD_SIZE: usize = BOARD_WIDTH * BOARD_HEIGHT;
pub const WALL_BOARD_SIZE: usize = (BOARD_WIDTH - 1) * (BOARD_HEIGHT - 1);
pub const OUTPUT_SIZE: usize = PAWN_BOARD_SIZE + WALL_BOARD_SIZE * 2;
pub const ASCII_LETTER_A: u8 = 97;
