use super::{Action, GameState, Predictions};
use engine::{engine::GameEngine, PlayerResult, PlayerScore, Players, ValidActions, Value};

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
        game_state
            .is_terminal()
            .map(|value| Predictions::new(value, game_state.victory_margin() as f32, game_state.move_number() as f32))
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

    fn valid_actions(&self, game_state: &Self::State) -> impl Iterator<Item = Self::Action> {
        game_state.valid_actions()
    }
}

impl Players for Engine {
    type State = GameState;

    fn players(&self, _state: &Self::State) -> &[usize] {
        static PLAYERS: [usize; 2] = [1, 2];
        &PLAYERS
    }
}

impl PlayerScore for Engine {
    type State = GameState;
    type PlayerScore = usize;

    fn score(&self, state: &Self::State, player_id: usize) -> Option<Self::PlayerScore> {
        state.is_terminal().map(|value| {
            if value.get_value_for_player(player_id) > 0.5 {
                state.victory_margin() as usize
            } else {
                0
            }
        })
    }
}

impl PlayerResult for Engine {
    type State = GameState;
    type PlayerResult = &'static str;

    fn result(&self, state: &Self::State, player_id: usize) -> Option<Self::PlayerResult> {
        state.is_terminal().map(|value| {
            match value.get_value_for_player(player_id) {
                v if v > 0.5 => "win",
                v if v < 0.5 => "loss",
                _ => "draw",
            }
        })
    }
}
