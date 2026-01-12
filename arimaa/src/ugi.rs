use std::borrow::Cow;
#[cfg(feature = "model")]
use std::path::PathBuf;

#[cfg(feature = "model")]
use crate::ModelFactory;
use crate::{Action, Engine, GameState};
use arimaa_engine::{Direction, Piece, Square, convert_piece_to_letter};
use engine::GameState as GameStateTrait;

use anyhow::Result;
use itertools::Itertools;
use ugi::{ActionsToMoveString, ConvertToValidCompositeActions, InitialGameState, MoveStringToActions, ParseGameState};

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

    fn actions_to_move_string(&self, game_state: &GameState, actions: &[Action]) -> String {
        convert_actions_to_move_string(game_state.clone(), actions)
    }
}

impl MoveStringToActions for UGI {
    type Action = Action;

    fn move_string_to_actions(&self, str: &str) -> Result<Vec<Action>> {
        convert_move_string_to_step_actions(str)
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

// use anyhow::Result;
// use arimaa::value::Value;
// use arimaa::{convert_piece_to_letter, Action, Direction, ModelRef, Piece, Square};
// use arimaa::{Analyzer, Engine, GameState, ModelFactory};
// use clap::Parser;
// use cli::Cli;
// use engine::engine::GameEngine as EngineTrait;
// use engine::game_state::GameState as GameStateTrait;
// use env_logger::Env;
// use itertools::Itertools;
// use mcts::mcts::MCTSNode;
// use mcts::mcts::{MCTSOptions, MCTS};
// use mcts::node_details::PUCT;
// use model::{Analyzer as AnalyzerTrait, Load};
// // use model::{Model as ModelTrait, ModelFactory as ModelFactoryTrait};
// use options::{UGIOption, UGIOptions};
// use rand::seq::IteratorRandom;
// use rand::thread_rng;
// use regex::Regex;
// use std::borrow::Cow;
// use std::io::stdin;
// use std::path::PathBuf;
// use std::sync::atomic::{AtomicBool, Ordering};
// use std::sync::{Arc, Mutex};
// use std::thread;
// use std::time::{Duration, Instant};
// use tokio::sync::mpsc;

// struct ArimaaUGI {}

// struct ArimaaUGIAction {}

// fn calculate_search_duration(
//     options: &UGIOptions,
//     game_state: &GameState,
//     current_player: usize,
// ) -> Duration {
//     let current_g_reserve_time = options.current_g_reserve_time;
//     let current_s_reserve_time = options.current_s_reserve_time;
//     let reserve_time_to_use = options.reserve_time_to_use;
//     let time_per_move = options.time_per_move;
//     let fixed_time = options.fixed_time;
//     let time_buffer = options.time_buffer;

//     let reserve_time: f32 = if current_player == 1 {
//         current_g_reserve_time
//     } else {
//         current_s_reserve_time
//     };
//     let reserve_time: f32 = reserve_time.min(reserve_time - time_per_move).max(0.0);
//     let search_time: f32 = reserve_time * reserve_time_to_use + time_per_move;
//     let search_time = search_time - time_buffer - time_per_move * 0.05;
//     let search_time: f32 = fixed_time.unwrap_or(search_time);
//     let search_time = if game_state.is_play_phase() {
//         search_time
//     } else {
//         search_time / 8.0
//     };

//     std::time::Duration::from_secs_f32(0f32.max(search_time))
// }

fn convert_actions_to_move_string(game_state: GameState, actions: &[Action]) -> String {
    let mut game_state = game_state;
    let mut actions_as_string = Vec::new();
    let actions_last_idx = actions.len() - 1;

    add_placements(&game_state, actions, &mut actions_as_string);

    for (i, action) in actions.iter().enumerate() {
        match action {
            Action::Move(_, _) | Action::PushPull(_, _) => {
                let mut tracking_game_state = Cow::Borrowed(&game_state);
                for (square, direction) in action.get_steps() {
                    let piece_board = tracking_game_state.get_piece_board();

                    let piece = &piece_board.get_piece_type_at_square(&square).unwrap();
                    let is_p1_piece =
                        piece_board.get_bits_for_piece(*piece, true) & square.as_bit_board() != 0;

                    actions_as_string.push(format!(
                        "{}{}{}",
                        convert_piece_to_letter(piece, is_p1_piece),
                        square,
                        direction
                    ));

                    let trapped_animal_square =
                        &tracking_game_state.get_trapped_animal_for_move(square, direction);
                    if let Some((square, piece, is_p1_piece)) = trapped_animal_square {
                        actions_as_string.push(format!(
                            "{}{}x",
                            convert_piece_to_letter(piece, *is_p1_piece),
                            square
                        ));
                    }

                    tracking_game_state =
                        Cow::Owned((*tracking_game_state).take_action(&Action::Move(
                            square,
                            std::iter::once(direction).collect(),
                        )));
                }
            }
            Action::Place(_) => {}
            Action::Pass => {}
        }

        let was_p1_move = game_state.is_p1_turn_to_move();
        game_state = game_state.take_action(action);
        let is_p1_move = game_state.is_p1_turn_to_move();

        // Double space move strings when switching between players
        if i != actions_last_idx && is_p1_move != was_p1_move {
            actions_as_string.push("".to_string());
        }
    }

    actions_as_string.iter().join(" ")
}

fn add_placements(game_state: &GameState, actions: &[Action], actions_string: &mut Vec<String>) {

    let rows = if game_state.is_p1_turn_to_move() {
        [2, 1]
    } else {
        [7, 8]
    };

    let all_placement_squares: Vec<_> = rows
        .iter()
        .cartesian_product('a'..='h')
        .map(|(row, col)| Square::new(col, *row))
        .collect();

    // Extract placed squares from actions and pair with their piece types based on order
    let placed_squares: Vec<_> = actions
        .iter()
        .take(8)
        .enumerate()
        .filter_map(|(i, action)| match action {
            Action::Place(square) => {
                let piece = match i {
                    0 => Piece::Elephant,
                    1 => Piece::Camel,
                    2..=3 => Piece::Horse,
                    4..=5 => Piece::Dog,
                    6..=7 => Piece::Cat,
                    _ => unreachable!(),
                };
                Some((*square, piece))
            }
            _ => None,
        })
        .collect();

    // Output all placement squares in order
    for square in all_placement_squares {
        let piece = placed_squares
            .iter()
            .find(|(s, _)| *s == square)
            .map(|(_, p)| *p)
            .unwrap_or(Piece::Rabbit);

        actions_string.push(format!(
            "{}{}",
            convert_piece_to_letter(&piece, game_state.is_p1_turn_to_move()),
            square
        ));
    }
}

fn convert_move_string_to_step_actions(actions_as_string: &str) -> Result<Vec<Action>> {
    let actions = actions_as_string
        .split(' ')
        .filter(|s| !s.contains('x'))
        .collect::<Vec<_>>();

    if actions.len() > 4 {
        let mut placements: Vec<(Piece, Square)> = actions
            .iter()
            .map(|s| {
                let square = s[1..=2].to_string().parse::<Square>().unwrap();
                let piece = s[0..1].to_string().parse::<Piece>().unwrap();
                (piece, square)
            })
            .filter(|(piece, _)| *piece != Piece::Rabbit)
            .collect();

        // Sort by piece type to match expected order: Elephant, Camel, Horse, Horse, Dog, Dog, Cat, Cat
        placements.sort_by_key(|(piece, _)| std::cmp::Reverse(*piece));

        // Return actions in the sorted order
        Ok(placements
            .into_iter()
            .map(|(_, square)| Action::Place(square))
            .collect())
            
    } else {
        let mut actions = actions
            .iter()
            .map(|s| {
                let square = s[1..3].parse::<Square>().unwrap();
                let direction = s[3..4].parse::<Direction>().unwrap();
                Action::Move(square, std::iter::once(direction).collect())
            })
            .collect::<Vec<_>>();

        if actions.len() <= 3 {
            actions.push(Action::Pass);
        }

        Ok(actions)
    }
}

// fn convert_to_valid_composite_actions(
//     step_actions: &[StepAction],
//     game_state: &GameState,
// ) -> Vec<Action> {
//     let composite_actions = convert_step_actions_into_composite_actions(step_actions);

//     if !game_state.is_play_phase() {
//         return composite_actions;
//     }

//     let target_game_state = composite_actions
//         .into_iter()
//         .fold(game_state.clone(), |target_game_state, a| {
//             target_game_state.take_action(&a)
//         });

//     let target_hash = target_game_state.get_transposition_hash();

//     match_target_hash(Cow::Borrowed(game_state), target_hash)
//         .expect("No valid set of actions found")
// }

// fn match_target_hash(game_state: Cow<GameState>, target_hash: u64) -> Option<Vec<Action>> {
//     let p1_turn_to_move = game_state.is_p1_turn_to_move();
//     for action in game_state.valid_actions() {
//         let new_game_state = game_state.take_action(&action);

//         if new_game_state.get_transposition_hash() == target_hash {
//             return Some(vec![action]);
//         }

//         if matches!(new_game_state.is_terminal(), Some(_))
//             || new_game_state.is_p1_turn_to_move() != p1_turn_to_move
//         {
//             continue;
//         }

//         if let Some(mut actions) = match_target_hash(Cow::Owned(new_game_state), target_hash) {
//             actions.insert(0, action);
//             return Some(actions);
//         }
//     }

//     None
// }

// fn convert_step_actions_into_composite_actions(step_actions: &[StepAction]) -> Vec<Action> {
//     let place_actions = step_actions
//         .iter()
//         .filter_map(|a| match a {
//             StepAction::Place(square, piece) => Some((square, piece)),
//             _ => None,
//         })
//         .filter(|(_, p)| **p != Piece::Rabbit)
//         .sorted_by(|(_, piece_a), (_, piece_b)| Ord::cmp(piece_b, piece_a))
//         .map(|(square, _)| Action::Place(*square));

//     let moves = step_actions.iter().filter_map(|a| match a {
//         StepAction::Place(_, _) => None,
//         StepAction::Pass => Some(Action::Pass),
//         StepAction::Move(square, dir) => {
//             Some(Action::Move(*square, std::iter::once(dir).collect()))
//         }
//     });

//     let actions = place_actions.chain(moves).collect();

//     log_debug(&format!("{:?}", actions));

//     actions
// }
