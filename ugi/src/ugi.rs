use anyhow::Result;

pub trait ActionsToMoveString {
    type State;
    type Action;

    fn actions_to_move_string(&self, game_state: &Self::State, actions: &[Self::Action]) -> String;
}

pub trait MoveStringToActions {
    type Action;

    fn move_string_to_actions(&self, str: &str) -> Result<Vec<Self::Action>>;
}

pub trait ParseGameState {
    type State;

    fn parse_game_state(&self, str: &str) -> Result<Self::State>;
}

pub trait InitialGameState {
    type State;

    fn initial_game_state(&self) -> Self::State;
}

/// Trait for converting actions to valid composite actions
/// This is useful for games like Arimaa where multiple simple actions
/// can be combined into composite actions with transposition handling
pub trait ConvertToValidCompositeActions {
    type State;
    type Action;

    fn convert_to_valid_composite_actions(
        &self,
        actions: &[Self::Action],
        game_state: &Self::State,
    ) -> Vec<Self::Action>;
}
