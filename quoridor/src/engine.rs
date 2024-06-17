use super::{Action, GameState, Value};
use engine::{engine::GameEngine, ValidActions};

#[derive(Default)]
pub struct Engine {}

impl Engine {
    pub fn new() -> Self {
        Self {}
    }
}

impl GameEngine for Engine {
    type Action = Action;
    type State = GameState;
    type Terminal = Predictions;

    fn take_action(&self, game_state: &Self::State, action: &Self::Action) -> Self::State {
        let mut game_state = game_state.clone();
        game_state.take_action(action);
        game_state
    }

    fn terminal_state(&self, game_state: &Self::State) -> Option<Self::Terminal> {
        game_state.is_terminal()
    }

    fn player_to_move(&self, game_state: &Self::State) -> usize {
        game_state.player_to_move()
    }

    fn move_number(&self, game_state: &Self::State) -> usize {
        game_state.move_number()
    }
}

impl ValidActions for Engine {
    type Action = Action;
    type State = GameState;

    fn valid_actions(&self, game_state: &Self::State) -> Vec<Self::Action> {
        let valid_pawn_moves = game_state.valid_pawn_move_actions();
        let valid_vert_walls = game_state.valid_vertical_wall_actions();
        let valid_horiz_walls = game_state.valid_horizontal_wall_actions();
        let actions = valid_pawn_moves
            .chain(valid_vert_walls)
            .chain(valid_horiz_walls);

        actions.collect()
    }
}
