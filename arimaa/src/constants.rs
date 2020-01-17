pub const ASCII_LETTER_A: u8 = 97;

pub const MAX_NUMBER_OF_MOVES: usize = 512;
pub const BOARD_WIDTH: usize = 8;
pub const BOARD_HEIGHT: usize = 8;
pub const BOARD_SIZE: usize = BOARD_WIDTH * BOARD_HEIGHT;

pub const NUM_PIECE_TYPES: usize = 6;
pub const NUM_PLAYERS: usize = 2;
pub const BOARDS_PER_STATE: usize = NUM_PIECE_TYPES * 2;
pub const MAX_NUM_STEPS: usize = 4;

pub const STEP_NUM_CHANNEL: usize = 1;
pub const STEP_NUM_CHANNEL_IDX: usize = BOARDS_PER_STATE * MAX_NUM_STEPS;
pub const TRAP_CHANNEL: usize = 1;
pub const TRAP_CHANNEL_IDX: usize = STEP_NUM_CHANNEL_IDX + 1;
pub const PLAY_INPUT_H: usize = BOARD_HEIGHT;
pub const PLAY_INPUT_W: usize = BOARD_WIDTH;
pub const PLAY_INPUT_C: usize = BOARDS_PER_STATE * MAX_NUM_STEPS + STEP_NUM_CHANNEL + TRAP_CHANNEL;
pub const PLAY_INPUT_SIZE: usize = BOARD_SIZE * PLAY_INPUT_C;
pub const PLAY_MOVES_LEFT_SIZE: usize = 128;

pub const PLACEMENT_BIT_CHANNEL: usize = 1;
pub const PLAYER_CHANNEL: usize = 1;
pub const PLACE_INPUT_H: usize = 4;
pub const PLACE_INPUT_W: usize = BOARD_WIDTH;
pub const PLACE_INPUT_C: usize = NUM_PIECE_TYPES + NUM_PIECE_TYPES + PLACEMENT_BIT_CHANNEL + PLAYER_CHANNEL;
pub const PLACE_INPUT_SIZE: usize = PLACE_BOARD_SIZE * PLACE_INPUT_C;
pub const PLACE_BOARD_SIZE: usize = PLACE_INPUT_W * PLACE_INPUT_H;
pub const PLACE_OUTPUT_SIZE: usize = NUM_PIECE_TYPES;
pub const PLACE_MOVES_LEFT_SIZE: usize = 1;

pub const NUM_UP_MOVES: usize = BOARD_SIZE - BOARD_WIDTH;
pub const NUM_RIGHT_MOVES: usize = BOARD_SIZE - BOARD_HEIGHT;
pub const NUM_DOWN_MOVES: usize = BOARD_SIZE - BOARD_WIDTH;
pub const NUM_LEFT_MOVES: usize = BOARD_SIZE - BOARD_HEIGHT;
pub const PASS_MOVES: usize = 1;

pub const PLAY_OUTPUT_SIZE: usize = NUM_UP_MOVES + NUM_RIGHT_MOVES + NUM_DOWN_MOVES + NUM_LEFT_MOVES + PASS_MOVES;
