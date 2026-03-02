use puct::TrajectoryTerminal;
use std::collections::HashSet;

use crate::{Action, GameState, Predictions};

pub struct RepetitionTerminal;

impl TrajectoryTerminal<GameState> for RepetitionTerminal {
    type Action = Action;
    type Terminal = Predictions;

    fn terminal_for_trajectory(
        &self,
        _state: &GameState,
        _action: &Action,
        _visited: &HashSet<u64>,
    ) -> Option<Predictions> {
        // @TODO: Implement
        // if state.is_action_3rd_repetition(action) {
        //     let (p1, p2) = if state.is_p1_turn_to_move() {
        //         (0.0f32, 1.0f32)
        //     } else {
        //         (1.0f32, 0.0f32)
        //     };
        //     Some(Predictions::new(
        //         Value::new(p1, p2),
        //         state.get_move_number() as f32,
        //     ))
        // } else {
        //     None
        // }
        panic!("RepetitionTerminal is not yet implemented")
    }
}
