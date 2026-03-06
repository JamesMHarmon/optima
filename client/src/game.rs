//! Game-specific type re-exports based on enabled feature flags.
//!
//! Build with one of:
//! - `cargo build` (quoridor default)
//! - `cargo build --features arimaa`
//! - `cargo build --features connect4`

#[cfg(not(any(feature = "quoridor", feature = "arimaa", feature = "connect4")))]
compile_error!("No game feature selected. Build with: cargo build [--features arimaa|connect4]");

#[cfg(quoridor_game)]
pub use quoridor::{Engine, ModelFactory, ModelRef, UGI};

#[cfg(quoridor_game)]
pub use ugi::BaseTimeStrategy as TimeStrategy;

#[cfg(arimaa_game)]
pub use arimaa::{ArimaaTimeStrategy as TimeStrategy, Engine, ModelFactory, ModelRef, UGI};

#[cfg(connect4_game)]
pub use connect4::{Engine, ModelFactory, ModelRef, UGI};

#[cfg(connect4_game)]
pub use ugi::BaseTimeStrategy as TimeStrategy;
