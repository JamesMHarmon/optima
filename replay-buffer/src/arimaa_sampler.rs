use arimaa::{Action, GameState, Mapper, Value};
use arimaa::{INPUT_SIZE, MOVES_LEFT_SIZE, OUTPUT_SIZE};
use engine::{GameEngine, Value as ValueTrait};
use half::f16;
use model::{ActionWithPolicy, NodeMetrics};
use tensorflow_model::{Dimension, InputMap, Mode, PolicyMap, ValueMap};

use crate::q_mix::{QMix, ValueStore};

use super::sample::Sample;

pub struct ArimaaSampler {
    engine: arimaa::Engine,
    mapper: Mapper,
}

impl ArimaaSampler {
    pub fn new(_mode: Option<String>) -> Self {
        Self {
            engine: arimaa::Engine::new(),
            mapper: Mapper::new(),
        }
    }
}

impl Dimension for ArimaaSampler {
    fn dimensions(&self) -> [u64; 3] {
        self.mapper.dimensions()
    }
}

impl Sample for ArimaaSampler {
    type State = GameState;
    type Action = Action;
    type Value = Value;
    type ValueStore = ArimaaVStore;

    fn take_action(&self, game_state: &Self::State, action: &Self::Action) -> Self::State {
        self.engine.take_action(game_state, action)
    }

    fn move_number(&self, game_state: &Self::State) -> usize {
        self.engine.get_move_number(game_state)
    }

    fn symmetries(
        &self,
        metric: model::PositionMetrics<Self::State, Self::Action, Self::Value>,
    ) -> Vec<model::PositionMetrics<Self::State, Self::Action, Self::Value>> {
        arimaa::get_symmetries(metric)
    }

    fn moves_left_size(&self) -> usize {
        MOVES_LEFT_SIZE
    }

    fn policy_size(&self) -> usize {
        OUTPUT_SIZE
    }

    fn input_size(&self) -> usize {
        INPUT_SIZE
    }
}

impl InputMap<GameState> for ArimaaSampler {
    fn game_state_to_input(&self, game_state: &GameState, input: &mut [f16], mode: Mode) {
        self.mapper.game_state_to_input(game_state, input, mode)
    }
}

impl PolicyMap<GameState, Action, Value> for ArimaaSampler {
    fn policy_metrics_to_expected_output(
        &self,
        game_state: &GameState,
        metric: &NodeMetrics<Action, Value>,
    ) -> Vec<f32> {
        self.mapper
            .policy_metrics_to_expected_output(game_state, metric)
    }

    fn policy_to_valid_actions(&self, _: &GameState, _: &[f16]) -> Vec<ActionWithPolicy<Action>> {
        panic!("Not implemented")
    }
}

impl ValueMap<GameState, Value> for ArimaaSampler {
    fn map_value_to_value_output(&self, game_state: &GameState, value: &Value) -> f32 {
        self.mapper.map_value_to_value_output(game_state, value)
    }

    fn map_value_output_to_value(&self, _: &GameState, _: f32) -> Value {
        panic!("Not implemented")
    }
}

#[allow(non_snake_case)]
impl QMix<GameState, Value> for ArimaaSampler {
    fn mix_q(game_state: &GameState, value: &Value, q_mix: f32, Q: f32) -> Value {
        if q_mix == 0.0 {
            return value.clone();
        }

        let player_to_move = game_state.player_to_move();
        let player_value = value.get_value_for_player(player_to_move);

        assert!(
            (0.0..=1.0).contains(&player_value),
            "player_value must be between 0.0 and 1.0"
        );

        let mixed_value = ((1.0 - q_mix) * player_value) + (q_mix * Q);

        assert!(
            (0.0..=1.0).contains(&q_mix) && (0.0..=1.0).contains(&Q),
            "Q mix must be between 0.0 and 1.0"
        );

        let mut value = value.clone();

        value.update_players_value(mixed_value, player_to_move);

        value
    }
}

#[derive(Default)]
pub struct ArimaaVStore([Option<Value>; 2]);

#[allow(non_snake_case)]
impl ValueStore<GameState, Value> for ArimaaVStore {
    fn get_v_for_player(&self, game_state: &GameState) -> Option<&Value> {
        let player = game_state.player_to_move();
        self.0[player - 1].as_ref()
    }

    fn set_v_for_player(&mut self, game_state: &GameState, V: Value) {
        let player = game_state.player_to_move();
        self.0[player - 1] = Some(V);
    }
}
