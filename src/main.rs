use rand::{thread_rng};
use std::time::{Instant};
use engine::*;

mod engine;
mod mcts;

struct QuoridorEngine {}

impl mcts::GameEngine<GameState, Action> for QuoridorEngine {
    fn get_state_analysis(&self, _: &GameState) -> mcts::GameStateAnalysis<Action> {
        mcts::GameStateAnalysis::new(
            vec!(mcts::ActionWithPolicy::new(
                Action::MovePawn(1),
                0.0
            )),
            0.0
        )
    }

    fn take_action(&self, game_state: &GameState, _: &Action) -> GameState {
        game_state.move_pawn(1)
    }
}


fn main() {
    let game_engine = QuoridorEngine {};
    let game_state = engine::GameState::new();
    let mut mcts = mcts::MCTS::new(
        game_state,
        &game_engine,
        mcts::MCTSOptions::new(
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