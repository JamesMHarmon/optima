pub trait GameEngine<S, A> {
    fn take_action(&self, game_state: &S, action: &A) -> S;
}
