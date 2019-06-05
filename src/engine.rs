
use super::analysis::GameStateAnalysis;

pub trait GameEngine<S, A> {
    fn get_state_analysis(&self, game_state: &S) -> GameStateAnalysis<A>;
    fn take_action(&self, game_state: &S, action: &A) -> S;
}
