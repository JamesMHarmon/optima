use puct::TrajectoryTerminal;
use std::collections::HashSet;

use crate::{GameState, Predictions};

pub struct RepetitionTerminal;

impl TrajectoryTerminal for RepetitionTerminal {
    type State = GameState;
    type Terminal = Predictions;

    fn terminal_for_trajectory(
        &self,
        _state: &GameState,
        _visited: &HashSet<u64>,
    ) -> Option<Predictions> {
        // Standard terminal detection will catch 3rd-position repetitions, so we don't need to do anything here.
        None
    }
}
