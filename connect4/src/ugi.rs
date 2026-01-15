#[cfg(feature = "model")]
use std::path::PathBuf;

#[cfg(feature = "model")]
use crate::ModelFactory;
use crate::{Action, Engine, GameState};
use engine::GameState as GameStateTrait;

use anyhow::Result;
use itertools::Itertools;
use ugi::{
    ActionsToMoveString, ConvertToValidCompositeActions, InitialGameState, MoveStringToActions,
    ParseGameState,
};

pub struct UGI {}

impl UGI {
    pub fn new() -> Self {
        Self {}
    }

    #[cfg(feature = "model")]
    pub fn model_factory(&self, model_dir: PathBuf) -> ModelFactory {
        ModelFactory::new(model_dir)
    }

    pub fn engine(&self) -> Engine {
        Engine::new()
    }
}

impl Default for UGI {
    fn default() -> Self {
        Self::new()
    }
}

impl ActionsToMoveString for UGI {
    type State = GameState;
    type Action = Action;

    fn actions_to_move_string(&self, _: &GameState, actions: &[Action]) -> String {
        actions.iter().map(|a| a.to_string()).join(" ")
    }
}

impl MoveStringToActions for UGI {
    type Action = Action;

    fn move_string_to_actions(&self, str: &str) -> Result<Vec<Action>> {
        let actions = str.split(' ').filter(|s| !s.is_empty()).map(|s| s.parse());

        actions.collect()
    }
}

impl ParseGameState for UGI {
    type State = GameState;

    fn parse_game_state(&self, _str: &str) -> Result<GameState> {
        // Connect4 doesn't have a standard FEN format, so we just return initial state
        // A more complete implementation would parse a position string
        Ok(GameState::initial())
    }
}

impl InitialGameState for UGI {
    type State = GameState;

    fn initial_game_state(&self) -> GameState {
        GameState::initial()
    }
}

impl ConvertToValidCompositeActions for UGI {
    type State = GameState;
    type Action = Action;

    fn convert_to_valid_composite_actions(
        &self,
        actions: &[Action],
        _state: &GameState,
    ) -> Vec<Action> {
        actions.to_vec()
    }
}
