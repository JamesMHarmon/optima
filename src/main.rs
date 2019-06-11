extern crate quoridor;

use rand::prelude::{SeedableRng, StdRng};
use std::time::{Instant};
use quoridor::engine::{GameEngine};
use quoridor::analysis::{ActionWithPolicy,GameStateAnalysis};
use quoridor::mcts::{DirichletOptions,MCTS,MCTSOptions};
use quoridor::quoridor::engine::{GameState};
use quoridor::quoridor::action::{Action};

struct QuoridorEngine {}

impl GameEngine<GameState, Action> for QuoridorEngine {
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

    fn take_action(&self, game_state: &GameState, _: &Action) -> GameState {
        game_state.take_action(Action::MovePawn(1))
    }
}


fn main() {
    let game_engine = QuoridorEngine {};
    let game_state = GameState::new();
    let seed: [u8; 32] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32];
    let seedable_rng: StdRng = SeedableRng::from_seed(seed);
    let mut mcts = MCTS::new(
        game_state,
        &game_engine,
        MCTSOptions::new(
            Some(DirichletOptions {
                alpha: 0.3,
                epsilon: 0.25
            }),
            &|_,_| { 4.0 },
            &|_| { 1.0 },
            seedable_rng,
        )
    );


    let now = Instant::now();
    let res = mcts.get_next_action(800).unwrap();
    let time = now.elapsed().as_millis();

    println!("TIME: {}",time);

    println!("{:?}", res);
    println!("{:?}", mcts.get_next_action(1));
    println!("{:?}", mcts.get_next_action(800));
    println!("{:?}", mcts.get_next_action(800));
}