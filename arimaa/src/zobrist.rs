use super::engine::{GameState,PieceBoardState};
use super::action::{Piece,Square,map_bit_board_to_squares};
use super::zobrist_values::*;

#[derive(Hash, Eq, PartialEq, Clone, Copy, Debug)]
pub struct Zobrist {
    hash: u64
}

impl Zobrist {
    pub fn initial() -> Self {
        Zobrist { hash: INITIAL }
    }

    pub fn from_piece_board(piece_board: &PieceBoardState, is_p1_turn_to_move: bool, step_num: usize) -> Self {
        let mut hash = INITIAL;
        if !is_p1_turn_to_move {
            hash ^= PLAYER_TO_MOVE;
        }

        hash ^= STEP_VALUES[step_num];

        for is_p1 in [true, false].into_iter() {
            for piece in [Piece::Elephant, Piece::Camel, Piece::Horse, Piece::Dog, Piece::Cat, Piece::Rabbit].into_iter() {
                let piece_bits = piece_board.get_bits_for_piece(piece, *is_p1);
                for square in map_bit_board_to_squares(piece_bits) {
                    hash ^= get_piece_value(&square, piece, *is_p1);
                }
            }
        }

        Zobrist { hash }
    }

    pub fn move_piece(&self, prev_game_state: &GameState, new_piece_board: &PieceBoardState, new_step: usize, new_p1_turn_to_move: bool) -> Self {
        let player_to_move_value = if prev_game_state.is_p1_turn_to_move() != new_p1_turn_to_move { PLAYER_TO_MOVE } else { 0 };
        let piece_board_value = get_piece_board_value(&prev_game_state.get_piece_board(), new_piece_board);
        let step_value = get_step_value(prev_game_state.get_current_step(), new_step);

        let hash = self.hash ^ player_to_move_value ^ piece_board_value ^ step_value;

        Zobrist { hash }
    }

    pub fn place_piece(&self, piece: &Piece, square: &Square, place_is_p1: bool, switch_players: bool, switch_phases: bool) -> Self {
        let player_to_move_value = if switch_players | switch_phases { PLAYER_TO_MOVE } else { 0 };
        let place_piece_value = get_piece_value(square, piece, place_is_p1);
        let step_value = if switch_phases { STEP_VALUES[0] } else { 0 };

        let hash = self.hash ^ player_to_move_value ^ place_piece_value ^ step_value;

        Zobrist { hash }
    }

    pub fn pass(&self, step: usize) -> Self {
        let hash = self.hash ^ PLAYER_TO_MOVE ^ STEP_VALUES[0] ^ STEP_VALUES[step];

        Zobrist { hash }
    }

    pub fn exclude_step(&self, step: usize) -> Self {
        let hash = self.hash ^ STEP_VALUES[0] ^ STEP_VALUES[step];

        Zobrist { hash }
    }

    pub fn hash(&self) -> u64 {
        self.hash
    }
}

fn get_step_value(prev_step: usize, new_step: usize) -> u64 {
    STEP_VALUES[prev_step] ^ STEP_VALUES[new_step]
}

fn get_piece_board_value(prev_piece_board: &PieceBoardState, new_piece_board: &PieceBoardState) -> u64 {
    let mut value = 0;

    for is_p1 in [true, false].into_iter() {
        for piece in [Piece::Elephant, Piece::Camel, Piece::Horse, Piece::Dog, Piece::Cat, Piece::Rabbit].into_iter() {
            let prev_piece_bits = prev_piece_board.get_bits_for_piece(piece, *is_p1);
            let new_piece_bits = new_piece_board.get_bits_for_piece(piece, *is_p1);
            let diff_bits = prev_piece_bits ^ new_piece_bits;

            if diff_bits != 0 {
                for square in map_bit_board_to_squares(diff_bits) {
                    value ^= get_piece_value(&square, piece, *is_p1);
                }
            }
        }
    }

    value
}

fn get_piece_value(square: &Square, piece: &Piece, is_p1: bool) -> u64 {
    let piece_idx = match piece {
        Piece::Elephant => 0,
        Piece::Camel => 1,
        Piece::Horse => 2,
        Piece::Dog => 3,
        Piece::Cat => 4,
        Piece::Rabbit => 5
    };

    let piece_idx = piece_idx + if is_p1 { 0 } else { 6 };

    SQUARE_VALUES[piece_idx][square.get_index()]
}