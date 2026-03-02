use puct::TrajectoryTerminal;
use std::collections::HashSet;

use crate::{GameState, Predictions, Value};

pub struct RepetitionTerminal;

impl TrajectoryTerminal for RepetitionTerminal {
    type State = GameState;
    type Terminal = Predictions;

    fn terminal_for_trajectory(
        &self,
        state: &GameState,
        visited: &HashSet<u64>,
    ) -> Option<Predictions> {
        let hash = state.transposition_hash();

        if visited.contains(&hash) {
            Some(Predictions::new(
                Value::new(0.5, 0.5),
                0.0,
                state.move_number() as f32,
            ))
        } else {
            None
        }
    }
}
