use std::path::PathBuf;

use crate::{Action, Engine, GameState, ModelFactory};
use engine::GameState as GameStateTrait;

use anyhow::Result;
use itertools::Itertools;
use ugi::{ActionsToMoveString, InitialGameState, MoveStringToActions, ParseGameState};

pub struct UGI {}

impl UGI {
    pub fn new() -> Self {
        Self {}
    }

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

    fn parse_game_state(&self, _: &str) -> GameState {
        todo!()
    }
}

impl InitialGameState for UGI {
    type State = GameState;

    fn initial_game_state(&self) -> GameState {
        GameState::initial()
    }
}
