use super::value::Value;

pub trait GameEngine {
    type Action;
    type State;
    type Value: Value;

    fn take_action(&self, game_state: &Self::State, action: &Self::Action) -> Self::State;
    fn player_to_move(&self, game_state: &Self::State) -> usize;
    fn move_number(&self, game_state: &Self::State) -> usize;
    fn terminal_state(&self, game_state: &Self::State) -> Option<Self::Value>;
}

pub trait ValidActions {
    type Action;
    type State;

    fn valid_actions(&self, game_state: &Self::State) -> Vec<Self::Action>;
}
