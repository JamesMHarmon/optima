use half::f16;
use tensorflow_model::transposition_entry::TranspositionEntry;

use super::constants::{PLACE_OUTPUT_SIZE, PLAY_OUTPUT_SIZE};

pub type PlayTranspositionEntry = TranspositionEntry<[f16; PLAY_OUTPUT_SIZE]>;
pub type PlaceTranspositionEntry = TranspositionEntry<[f16; PLACE_OUTPUT_SIZE]>;
