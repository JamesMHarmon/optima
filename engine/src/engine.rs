pub trait GameEngine {
    type Action;
    type State;

    fn take_action(&self, game_state: &Self::State, action: &Self::Action) -> Self::State;
    fn is_terminal_state(&self, game_state: &Self::State) -> Option<f64>;
}
