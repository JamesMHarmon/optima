extern crate quoridor;

use rand::{thread_rng};
use std::time::{Instant};
use quoridor::engine::{GameEngine};
use quoridor::analysis::{ActionWithPolicy,GameStateAnalysis};
use quoridor::mcts::{MCTS,MCTSOptions};
use quoridor::quoridor::engine::{GameState};
use quoridor::quoridor::action::{Action};

struct QuoridorEngine {}

impl GameEngine<GameState, Action> for QuoridorEngine {
    fn get_state_analysis(&self, _: &GameState) -> GameStateAnalysis<Action> {
        GameStateAnalysis::new(
            vec!(ActionWithPolicy::new(
                Action::MovePawn(1),
                0.0
            )),
            0.0
        )
    }

    fn take_action(&self, game_state: &GameState, _: &Action) -> GameState {
        game_state.take_action(Action::MovePawn(1))
    }
}


fn main() {
    let game_engine = QuoridorEngine {};
    let game_state = GameState::new();
    let mut mcts = MCTS::new(
        game_state,
        &game_engine,
        MCTSOptions::new(
            0.0,
            0.0,
            &|_| { 0.0 },
            &|_| { 0.0 },
            thread_rng(),
        )
    );

    let now = Instant::now();
    let res = mcts.get_next_action(80000);
    let time = now.elapsed().as_millis();
    println!("TIME: {}",time);

    println!("{:?}", res);
    println!("{:?}", mcts.get_next_action(800));
    println!("{:?}", mcts.get_next_action(800));
    println!("{:?}", mcts.get_next_action(800));
}