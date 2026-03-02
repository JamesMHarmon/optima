use puct::TrajectoryTerminal;
use std::collections::HashSet;

use crate::{Action, GameState, Predictions, Value};

pub struct RepetitionTerminal;

impl TrajectoryTerminal<GameState> for RepetitionTerminal {
    type Action = Action;
    type Terminal = Predictions;

    fn terminal_for_trajectory(
        &self,
        state: &GameState,
        action: &Action,
        visited: &HashSet<u64>,
    ) -> Option<Predictions> {
        let mut state = state.clone();
        state.take_action(action);
        let next_hash = state.transposition_hash();

        if visited.contains(&next_hash) {
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
