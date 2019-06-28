use super::super::analytics::{ActionWithPolicy,GameAnalytics,GameStateAnalysis};
use super::engine::{GameState};
use super::action::{Action};
use super::engine::Engine;

impl GameAnalytics for Engine {
    type Action = Action;
    type State = GameState;

    fn get_state_analysis(&self, _: &GameState) -> GameStateAnalysis<Action> {
        GameStateAnalysis::new(
            vec!(ActionWithPolicy::new(
                Action::MovePawn(1),
                0.1
            ), ActionWithPolicy::new(
                Action::MovePawn(2),
                0.2
            ), ActionWithPolicy::new(
                Action::MovePawn(3),
                0.32
            ), ActionWithPolicy::new(
                Action::MovePawn(4),
                0.1
            ), ActionWithPolicy::new(
                Action::MovePawn(5),
                0.30
            )),
            0.5
        )
    }
}