pub trait GameEngine<S, A> {
    fn take_action(&self, game_state: &S, action: &A) -> S;
    fn is_terminal_state(&self, game_state: &S) -> Option<f64>;
}
