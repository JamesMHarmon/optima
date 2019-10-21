pub const ASCII_LETTER_A: u8 = 97;

pub const MAX_NUMBER_OF_MOVES: usize = 128;
pub const BOARD_WIDTH: usize = 8;
pub const BOARD_HEIGHT: usize = 8;


pub const PLAY_INPUT_H: usize = BOARD_HEIGHT;
pub const PLAY_INPUT_W: usize = BOARD_WIDTH;
pub const NUM_PIECE_TYPES: usize = 6;
pub const NUM_PLAYERS: usize = 2;
pub const BOARDS_PER_STATE: usize = NUM_PIECE_TYPES * 2;
pub const MAX_NUM_STATES_PER_MOVE: usize = 4;
pub const PLAY_INPUT_C: usize = BOARDS_PER_STATE * MAX_NUM_STATES_PER_MOVE;
pub const PLAY_BOARD_SIZE: usize = BOARD_WIDTH * BOARD_HEIGHT;

pub const NUM_UP_MOVES: usize = PLAY_BOARD_SIZE - BOARD_WIDTH;
pub const NUM_RIGHT_MOVES: usize = PLAY_BOARD_SIZE - BOARD_HEIGHT;
pub const NUM_DOWN_MOVES: usize = PLAY_BOARD_SIZE - BOARD_WIDTH;
pub const NUM_LEFT_MOVES: usize = PLAY_BOARD_SIZE - BOARD_HEIGHT;
pub const PASS_MOVES: usize = 1;

pub const PLAY_OUTPUT_SIZE: usize = NUM_UP_MOVES + NUM_RIGHT_MOVES + NUM_DOWN_MOVES + NUM_LEFT_MOVES + PASS_MOVES;
