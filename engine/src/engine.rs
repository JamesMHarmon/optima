use super::value::Value;

pub trait GameEngine {
    type Action;
    type State;
    type Value: Value;

    fn take_action(&self, game_state: &Self::State, action: &Self::Action) -> Self::State;
    fn get_player_to_move(&self, game_state: &Self::State) -> usize;
    fn get_move_number(&self, game_state: &Self::State) -> usize;
    fn is_terminal_state(&self, game_state: &Self::State) -> Option<Self::Value>;
}
