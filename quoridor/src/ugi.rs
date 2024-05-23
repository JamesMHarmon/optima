use std::path::PathBuf;

use crate::{Action, Coordinate, Engine, GameState, ModelFactory};
use engine::GameState as GameStateTrait;

use anyhow::{anyhow, Result};
use itertools::Itertools;
use regex::Regex;
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

    // Parses a FEN state. d4f4e7 / a2a8 / e4 e6 / 7 8 / 2
    fn parse_game_state(&self, str: &str) -> Result<GameState> {
        let coord_parts_re = Regex::new(r"[a-z][1-9]").unwrap();
        let walls_remaining_re = Regex::new(r"[0-9]+").unwrap();
        let mut fen_parts = str.split('/');

        let horizontal_walls = fen_parts
            .next()
            .and_then(|s| {
                coord_parts_re
                    .find_iter(s)
                    .map(|c| c.as_str().parse::<Coordinate>().ok())
                    .collect::<Option<Vec<_>>>()
            })
            .ok_or(anyhow!("Invalid FEN format"))?;
        let vertical_walls = fen_parts
            .next()
            .and_then(|s| {
                coord_parts_re
                    .find_iter(s)
                    .map(|c| c.as_str().parse::<Coordinate>().ok())
                    .collect::<Option<Vec<_>>>()
            })
            .ok_or(anyhow!("Invalid FEN format"))?;
        let player_positions = fen_parts
            .next()
            .and_then(|s| {
                coord_parts_re
                    .find_iter(s)
                    .map(|c| c.as_str().parse::<Coordinate>().ok())
                    .collect::<Option<Vec<_>>>()
            })
            .ok_or(anyhow!("Invalid FEN format"))?;
        let walls_remaining = fen_parts
            .next()
            .and_then(|s| {
                walls_remaining_re
                    .find_iter(s)
                    .map(|c| c.as_str().parse::<usize>().ok())
                    .collect::<Option<Vec<_>>>()
            })
            .ok_or(anyhow!("Invalid FEN format"))?;
        let p1_turn_to_move = fen_parts
            .next()
            .and_then(|s| s.parse::<usize>().ok())
            .ok_or(anyhow!("Invalid FEN format"))?
            == 1;

        Ok(GameState::new(
            horizontal_walls,
            vertical_walls,
            player_positions,
            walls_remaining,
            p1_turn_to_move,
        ))
    }
}

impl InitialGameState for UGI {
    type State = GameState;

    fn initial_game_state(&self) -> GameState {
        GameState::initial()
    }
}
