use arimaa::place_model::Mapper as PlaceMapper;
use arimaa::play_model::Mapper as PlayMapper;
use arimaa::{
    Action, GameState, Value, PLACE_INPUT_SIZE, PLACE_MOVES_LEFT_SIZE, PLACE_OUTPUT_SIZE,
    PLAY_INPUT_SIZE, PLAY_MOVES_LEFT_SIZE, PLAY_OUTPUT_SIZE,
};
use engine::GameEngine;
use half::f16;
use model::tensorflow::{Dimension, InputMap, Mode, PolicyMap, ValueMap};
use model::{ActionWithPolicy, NodeMetrics};

use super::sample::Sample;

pub struct ArimaaSampler {
    is_play_mode: bool,
    engine: arimaa::Engine,
    play_model_mapper: PlayMapper,
    place_model_mapper: PlaceMapper,
}

impl ArimaaSampler {
    pub fn new(mode: Option<String>) -> Self {
        let mode = mode.expect("Mode must be defined for ArimaaSampler");

        let is_play_mode = if mode == "play" {
            true
        } else if mode == "place" {
            false
        } else {
            panic!("Expected mode to be play or place")
        };

        Self {
            is_play_mode,
            engine: arimaa::Engine::new(),
            play_model_mapper: PlayMapper::new(),
            place_model_mapper: PlaceMapper::new(),
        }
    }
}

impl Dimension for ArimaaSampler {
    fn dimensions(&self) -> [u64; 3] {
        if self.is_play_mode {
            self.play_model_mapper.dimensions()
        } else {
            self.place_model_mapper.dimensions()
        }
    }
}

impl Sample for ArimaaSampler {
    type State = GameState;
    type Action = Action;
    type Value = Value;

    fn take_action(&self, game_state: &Self::State, action: &Self::Action) -> Self::State {
        self.engine.take_action(game_state, action)
    }

    fn move_number(&self, game_state: &Self::State) -> usize {
        self.engine.get_move_number(game_state)
    }

    fn sample_filter(
        &self,
        metric: &model::PositionMetrics<Self::State, Self::Action, Self::Value>,
    ) -> bool {
        self.is_play_mode ^ !metric.game_state.is_play_phase()
    }

    fn symmetries(
        &self,
        metric: model::PositionMetrics<Self::State, Self::Action, Self::Value>,
    ) -> Vec<model::PositionMetrics<Self::State, Self::Action, Self::Value>> {
        arimaa::get_symmetries(metric)
    }

    fn moves_left_size(&self) -> usize {
        if self.is_play_mode {
            PLAY_MOVES_LEFT_SIZE
        } else {
            PLACE_MOVES_LEFT_SIZE
        }
    }

    fn policy_size(&self) -> usize {
        if self.is_play_mode {
            PLAY_OUTPUT_SIZE
        } else {
            PLACE_OUTPUT_SIZE
        }
    }

    fn input_size(&self) -> usize {
        if self.is_play_mode {
            PLAY_INPUT_SIZE
        } else {
            PLACE_INPUT_SIZE
        }
    }
}

impl InputMap<GameState> for ArimaaSampler {
    fn game_state_to_input(&self, game_state: &GameState, mode: Mode) -> Vec<f16> {
        if self.is_play_mode {
            self.play_model_mapper.game_state_to_input(game_state, mode)
        } else {
            self.place_model_mapper
                .game_state_to_input(game_state, mode)
        }
    }
}

impl PolicyMap<GameState, Action> for ArimaaSampler {
    fn policy_metrics_to_expected_output(
        &self,
        game_state: &GameState,
        metric: &NodeMetrics<Action>,
    ) -> Vec<f32> {
        if self.is_play_mode {
            self.play_model_mapper
                .policy_metrics_to_expected_output(game_state, metric)
        } else {
            self.place_model_mapper
                .policy_metrics_to_expected_output(game_state, metric)
        }
    }

    fn policy_to_valid_actions(&self, _: &GameState, _: &[f16]) -> Vec<ActionWithPolicy<Action>> {
        panic!("Not implemented")
    }
}

impl ValueMap<GameState, Value> for ArimaaSampler {
    fn map_value_to_value_output(&self, game_state: &GameState, value: &Value) -> f32 {
        if self.is_play_mode {
            self.play_model_mapper
                .map_value_to_value_output(game_state, value)
        } else {
            self.place_model_mapper
                .map_value_to_value_output(game_state, value)
        }
    }

    fn map_value_output_to_value(&self, _: &GameState, _: f32) -> Value {
        panic!("Not implemented")
    }
}
