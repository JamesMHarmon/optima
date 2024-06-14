use half::f16;
use tensorflow_model::TranspositionEntry as TFTranspositionEntry;

use super::constants::OUTPUT_SIZE;

pub type TranspositionEntry = TFTranspositionEntry<[f16; OUTPUT_SIZE]>;
