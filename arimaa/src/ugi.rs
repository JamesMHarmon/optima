use anyhow::Result;
use arimaa::value::Value;
use arimaa::{convert_piece_to_letter, Action, Direction, ModelRef, Piece, Square};
use arimaa::{Analyzer, Engine, GameState, ModelFactory};
use clap::Parser;
use cli::Cli;
use engine::engine::GameEngine as EngineTrait;
use engine::game_state::GameState as GameStateTrait;
use env_logger::Env;
use itertools::Itertools;
use mcts::mcts::MCTSNode;
use mcts::mcts::{MCTSOptions, MCTS};
use mcts::node_details::PUCT;
use model::{Analyzer as AnalyzerTrait, Load};
// use model::{Model as ModelTrait, ModelFactory as ModelFactoryTrait};
use options::{UGIOption, UGIOptions};
use rand::seq::IteratorRandom;
use rand::thread_rng;
use regex::Regex;
use std::borrow::Cow;
use std::io::stdin;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

struct ArimaaUGI {}

struct ArimaaUGIAction {}

fn calculate_search_duration(
    options: &UGIOptions,
    game_state: &GameState,
    current_player: usize,
) -> Duration {
    let current_g_reserve_time = options.current_g_reserve_time;
    let current_s_reserve_time = options.current_s_reserve_time;
    let reserve_time_to_use = options.reserve_time_to_use;
    let time_per_move = options.time_per_move;
    let fixed_time = options.fixed_time;
    let time_buffer = options.time_buffer;

    let reserve_time: f32 = if current_player == 1 {
        current_g_reserve_time
    } else {
        current_s_reserve_time
    };
    let reserve_time: f32 = reserve_time.min(reserve_time - time_per_move).max(0.0);
    let search_time: f32 = reserve_time * reserve_time_to_use + time_per_move;
    let search_time = search_time - time_buffer - time_per_move * 0.05;
    let search_time: f32 = fixed_time.unwrap_or(search_time);
    let search_time = if game_state.is_play_phase() {
        search_time
    } else {
        search_time / 8.0
    };

    std::time::Duration::from_secs_f32(0f32.max(search_time))
}

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
    if actions.iter().all(|a| !matches!(a, Action::Place(_))) {
        return;
    }

    assert_eq!(actions.len(), 8, "Placement actions should always be 8");

    let rows = if game_state.is_p1_turn_to_move() {
        [2, 1]
    } else {
        [7, 8]
    };

    let placement_squares: Vec<_> = rows
        .iter()
        .cartesian_product('a'..='h')
        .map(|(row, col)| Square::new(col, *row))
        .collect();

    for square in placement_squares {
        let pos = actions
            .iter()
            .find_position(|a| match a {
                Action::Place(s) => square == *s,
                _ => false,
            })
            .map(|(p, _)| p);

        let piece = match pos {
            Some(0) => Piece::Elephant,
            Some(1) => Piece::Camel,
            Some(2..=3) => Piece::Horse,
            Some(4..=5) => Piece::Dog,
            Some(6..=7) => Piece::Cat,
            _ => Piece::Rabbit,
        };

        actions_string.push(format!(
            "{}{}",
            convert_piece_to_letter(&piece, game_state.is_p1_turn_to_move()),
            square
        ));
    }
}

fn convert_move_string_to_step_actions(actions_as_string: &str) -> Vec<StepAction> {
    let actions = actions_as_string
        .split(' ')
        .filter(|s| !s.contains('x'))
        .collect::<Vec<_>>();

    if actions.len() > 4 {
        actions
            .iter()
            .map(|s| {
                let square = s[1..=2].to_string().parse::<Square>().unwrap();
                let piece = s[0..1].to_string().parse::<Piece>().unwrap();
                StepAction::Place(square, piece)
            })
            .collect::<Vec<_>>()
    } else {
        let mut actions = actions
            .iter()
            .map(|s| {
                let square = s[1..3].parse::<Square>().unwrap();
                let direction = s[3..4].parse::<Direction>().unwrap();
                StepAction::Move(square, direction)
            })
            .collect::<Vec<_>>();

        if actions.len() <= 3 {
            actions.push(StepAction::Pass);
        }

        actions
    }
}

fn mirror_actions(actions: &[Action], reflective_symmetry: bool) -> Vec<Action> {
    actions
        .iter()
        .map(|action| match action {
            Action::Place(_) | Action::Move(_, _) | Action::PushPull(_, _) | Action::Pass => {
                let rotated_action = action.invert();
                if reflective_symmetry {
                    rotated_action.vertical_symmetry()
                } else {
                    rotated_action
                }
            }
        })
        .collect()
}

fn is_setup_terminating_actions(actions: &[Action]) -> bool {
    let num_move_steps: usize = actions
        .iter()
        .filter(|action| matches!(action, Action::Move(_, _)))
        .map(|action| action.get_steps().len())
        .sum();

    num_move_steps > 0 && num_move_steps <= 2
}

fn find_transpositions<C, T>(
    actions: &[Action],
    game_state: &GameState,
    mcts: &MCTS<GameState, Action, Engine, Analyzer, C, T, Value>,
) -> Vec<(Vec<Action>, usize)>
where
    C: Fn(&GameState, usize, bool) -> f32,
    T: Fn(&GameState) -> f32,
{
    mcts.get_root_node()
        .ok()
        .map(|root_node| {
            let is_p1_turn_to_move = game_state.is_p1_turn_to_move();
            let target_hash = actions
                .iter()
                .fold(game_state.clone(), |game_state, action| {
                    game_state.take_action(action)
                })
                .get_transposition_hash();

            most_visited_transposition_actions(
                &root_node,
                game_state,
                is_p1_turn_to_move,
                target_hash,
                mcts,
            )
        })
        .unwrap_or_default()
}

fn most_visited_transposition_actions<C, T>(
    node: &MCTSNode<Action, Value>,
    game_state: &GameState,
    is_p1_turn_to_move: bool,
    target_hash: u64,
    mcts: &MCTS<GameState, Action, Engine, Analyzer, C, T, Value>,
) -> Vec<(Vec<Action>, usize)>
where
    C: Fn(&GameState, usize, bool) -> f32,
    T: Fn(&GameState) -> f32,
{
    let mut transpositions = vec![];
    for edge in node.iter_edges() {
        if let Some(node) = mcts.get_node_of_edge(edge) {
            let game_state = game_state.take_action(edge.action());
            if game_state.get_transposition_hash() == target_hash {
                transpositions.push((vec![edge.action().clone()], edge.visits()));
                continue;
            }

            if node.is_terminal() || game_state.is_p1_turn_to_move() != is_p1_turn_to_move {
                continue;
            }

            let child_transpositions = most_visited_transposition_actions(
                &node,
                &game_state,
                is_p1_turn_to_move,
                target_hash,
                mcts,
            );

            for (mut actions, visits) in child_transpositions {
                actions.insert(0, edge.action().clone());
                transpositions.push((actions, visits));
            }
        }
    }

    transpositions
}

#[derive(Clone, Copy, Debug)]
enum StepAction {
    Place(Square, Piece),
    Move(Square, Direction),
    Pass,
}

fn convert_to_valid_composite_actions(
    step_actions: &[StepAction],
    game_state: &GameState,
) -> Vec<Action> {
    let composite_actions = convert_step_actions_into_composite_actions(step_actions);

    if !game_state.is_play_phase() {
        return composite_actions;
    }

    let target_game_state = composite_actions
        .into_iter()
        .fold(game_state.clone(), |target_game_state, a| {
            target_game_state.take_action(&a)
        });

    let target_hash = target_game_state.get_transposition_hash();

    match_target_hash(Cow::Borrowed(game_state), target_hash)
        .expect("No valid set of actions found")
}

fn match_target_hash(game_state: Cow<GameState>, target_hash: u64) -> Option<Vec<Action>> {
    let p1_turn_to_move = game_state.is_p1_turn_to_move();
    for action in game_state.valid_actions() {
        let new_game_state = game_state.take_action(&action);

        if new_game_state.get_transposition_hash() == target_hash {
            return Some(vec![action]);
        }

        if matches!(new_game_state.is_terminal(), Some(_))
            || new_game_state.is_p1_turn_to_move() != p1_turn_to_move
        {
            continue;
        }

        if let Some(mut actions) = match_target_hash(Cow::Owned(new_game_state), target_hash) {
            actions.insert(0, action);
            return Some(actions);
        }
    }

    None
}

fn convert_step_actions_into_composite_actions(step_actions: &[StepAction]) -> Vec<Action> {
    let place_actions = step_actions
        .iter()
        .filter_map(|a| match a {
            StepAction::Place(square, piece) => Some((square, piece)),
            _ => None,
        })
        .filter(|(_, p)| **p != Piece::Rabbit)
        .sorted_by(|(_, piece_a), (_, piece_b)| Ord::cmp(piece_b, piece_a))
        .map(|(square, _)| Action::Place(*square));

    let moves = step_actions.iter().filter_map(|a| match a {
        StepAction::Place(_, _) => None,
        StepAction::Pass => Some(Action::Pass),
        StepAction::Move(square, dir) => {
            Some(Action::Move(*square, std::iter::once(dir).collect()))
        }
    });

    let actions = place_actions.chain(moves).collect();

    log_debug(&format!("{:?}", actions));

    actions
}
