//! Game-specific type re-exports based on enabled feature flags.
//!
//! Build with one of:
//! - `cargo build --features quoridor` (default)
//! - `cargo build --features arimaa --no-default-features`
//! - `cargo build --features connect4 --no-default-features`

// Ensure exactly one game feature is enabled
#[cfg(all(feature = "quoridor", feature = "arimaa"))]
compile_error!("Features 'quoridor' and 'arimaa' are mutually exclusive. Choose one.");

#[cfg(all(feature = "quoridor", feature = "connect4"))]
compile_error!("Features 'quoridor' and 'connect4' are mutually exclusive. Choose one.");

#[cfg(all(feature = "arimaa", feature = "connect4"))]
compile_error!("Features 'arimaa' and 'connect4' are mutually exclusive. Choose one.");

#[cfg(not(any(feature = "quoridor", feature = "arimaa", feature = "connect4")))]
compile_error!("At least one game feature must be enabled: 'quoridor', 'arimaa', or 'connect4'.");

#[cfg(feature = "quoridor")]
pub use quoridor::{Engine, ModelFactory, ModelRef, UGI};

#[cfg(feature = "quoridor")]
pub use ugi::BaseTimeStrategy as TimeStrategy;

#[cfg(feature = "arimaa")]
pub use arimaa::{ArimaaTimeStrategy as TimeStrategy, Engine, ModelFactory, ModelRef, UGI};

#[cfg(feature = "connect4")]
pub use connect4::{Engine, ModelFactory, ModelRef, UGI};

#[cfg(feature = "connect4")]
pub use ugi::BaseTimeStrategy as TimeStrategy;
