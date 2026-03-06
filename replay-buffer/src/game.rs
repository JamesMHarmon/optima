#[cfg(not(any(feature = "quoridor", feature = "arimaa", feature = "connect4")))]
compile_error!("No game feature selected. Build with: cargo build [--features arimaa|connect4]");

#[cfg(quoridor_game)]
pub use crate::quoridor_sampler::QuoridorSampler as Sampler;

#[cfg(arimaa_game)]
pub use crate::arimaa_sampler::ArimaaSampler as Sampler;

#[cfg(connect4_game)]
pub use crate::connect4_sampler::Connect4Sampler as Sampler;
