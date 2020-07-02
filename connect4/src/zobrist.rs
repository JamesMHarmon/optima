use super::zobrist_values::*;

#[derive(Eq, PartialEq, Clone, Copy, Debug)]
pub struct Zobrist {
    hash: u64
}

impl Zobrist {
    pub fn initial() -> Self {
        Zobrist { hash: INITIAL }
    }

    pub fn add_piece(&self, piece_bit: u64, is_p1: bool) -> Self {
        let player_idx = if is_p1 { 0 } else { 1 };
        let bit_idx = piece_bit.trailing_zeros() as usize;
        let hash = self.hash ^ SQUARE_VALUES[player_idx][bit_idx];

        Zobrist { hash }  
    }

    pub fn board_state_hash(&self) -> u64 {
        self.hash
    }
}
