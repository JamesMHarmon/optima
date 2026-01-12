#[cfg(feature = "quoridor")]
pub use crate::quoridor_sampler::QuoridorSampler as Sampler;

#[cfg(feature = "arimaa")]
pub use crate::arimaa_sampler::ArimaaSampler as Sampler;

#[cfg(feature = "connect4")]
pub use crate::connect4_sampler::Connect4Sampler as Sampler;
