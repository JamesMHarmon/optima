pub use arimaa_engine::constants::*;

pub const MAX_NUMBER_OF_MOVES: usize = 256;

pub const BOARD_PIECES_NUM_CHANNELS: usize = NUM_PIECE_TYPES * 2;
pub const STEP_NUM_CHANNELS: usize = MAX_NUM_STEPS - 1;
pub const STEP_NUM_CHANNEL_IDX: usize = BOARD_PIECES_NUM_CHANNELS;
pub const BANNED_PIECES_CHANNEL_IDX: usize = STEP_NUM_CHANNEL_IDX + STEP_NUM_CHANNELS;
pub const BANNED_PIECES_NUM_CHANNELS: usize = 1;
pub const TRAP_NUM_CHANNELS: usize = 1;
pub const TRAP_CHANNEL_IDX: usize = BANNED_PIECES_CHANNEL_IDX + BANNED_PIECES_NUM_CHANNELS;
pub const INPUT_H: usize = BOARD_HEIGHT;
pub const INPUT_W: usize = BOARD_WIDTH;
pub const INPUT_C: usize =
    BOARD_PIECES_NUM_CHANNELS + STEP_NUM_CHANNELS + BANNED_PIECES_NUM_CHANNELS + TRAP_NUM_CHANNELS;
pub const INPUT_SIZE: usize = BOARD_SIZE * INPUT_C;
pub const MOVES_LEFT_SIZE: usize = 128;

pub const NUM_PIECE_MOVES: usize =
    // Single direction moves
    4 * (BOARD_WIDTH * (BOARD_HEIGHT - 1)) +
    // Double direction moves
    4 * (BOARD_WIDTH * (BOARD_HEIGHT - 2)) +
    // Bi direction two step moves
    4 * ((BOARD_WIDTH - 1) * (BOARD_HEIGHT -1)) +
    // Triple direction move
    4 * (BOARD_WIDTH * (BOARD_HEIGHT - 3)) +
    // Bi direction tri step moves
    8 * ((BOARD_WIDTH - 1) * (BOARD_HEIGHT - 2)) +
    // Quad direction move
    4 * (BOARD_WIDTH * (BOARD_HEIGHT - 4)) +
    // Bi direction bi direction four step moves
    4 * ((BOARD_WIDTH - 2) * (BOARD_HEIGHT - 2)) +
    // Tri direction uno direction four step moves
    8 * ((BOARD_WIDTH - 3) * (BOARD_HEIGHT - 1));

pub const NUM_PUSH_PULL_MOVES: usize =
    // Horizontal push pull moves
    2 * ((BOARD_WIDTH - 2) * BOARD_HEIGHT) +
    // Vertical push pull moves
    2 * (BOARD_WIDTH * (BOARD_HEIGHT - 2)) +
    // Perpendicular
    8 * ((BOARD_WIDTH - 1) * (BOARD_HEIGHT - 1));

pub const PASS_MOVES: usize = 1;

pub const OUTPUT_SIZE: usize = NUM_PIECE_MOVES + NUM_PUSH_PULL_MOVES + PASS_MOVES;
