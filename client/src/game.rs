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

// Quoridor exports
#[cfg(feature = "quoridor")]
pub use quoridor::{
    Engine, ModelFactory, ModelRef, QuoridorBackpropagationStrategy as BackpropagationStrategy,
    QuoridorSelectionStrategy as SelectionStrategy, QuoridorStrategyOptions as StrategyOptions,
    UGI,
};

// Arimaa exports
#[cfg(feature = "arimaa")]
compile_error!(
    "Arimaa support is not yet fully implemented. Missing: BackpropagationStrategy, SelectionStrategy, StrategyOptions, UGI. Use --features quoridor instead."
);

// Connect4 exports
#[cfg(feature = "connect4")]
pub use connect4::{
    Connect4BackpropagationStrategy as BackpropagationStrategy,
    Connect4SelectionStrategy as SelectionStrategy, Connect4StrategyOptions as StrategyOptions,
    Engine, ModelFactory, ModelRef, UGI,
};
