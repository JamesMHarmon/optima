use std::collections::HashMap;

use common::{MovesLeftPropagatedValue, PropagatedGameLength, PropagatedValue};
use connect4::{Action, GameState, Mapper, Predictions, Value};
use connect4::{INPUT_C, INPUT_H, INPUT_W, MOVES_LEFT_SIZE, OUTPUT_SIZE};
use engine::{GameEngine, Value as ValueTrait};
use half::f16;
use model::{NodeMetrics, PositionMetrics};
use tensorflow_model::{Dimension, InputMap, Mode, PredictionsMap};

use crate::q_mix::{PredictionStore, QMix};

use super::sample::Sample;

type Connect4PositionMetrics =
    PositionMetrics<GameState, Action, Predictions, MovesLeftPropagatedValue>;
type Connect4NodeMetrics = NodeMetrics<Action, Predictions, MovesLeftPropagatedValue>;
type OutputMap = HashMap<String, Vec<f32>>;

const INPUT_SIZE: usize = INPUT_H * INPUT_W * INPUT_C;

pub struct Connect4Sampler {
    engine: connect4::Engine,
    mapper: Mapper,
}

impl Connect4Sampler {
    pub fn new(_mode: Option<String>) -> Self {
        Self {
            engine: connect4::Engine::new(),
            mapper: Mapper::new(),
        }
    }
}

impl Dimension for Connect4Sampler {
    fn dimensions(&self) -> [u64; 3] {
        self.mapper.dimensions()
    }
}

impl Sample for Connect4Sampler {
    type State = GameState;
    type Action = Action;
    type Predictions = Predictions;
    type PropagatedValues = MovesLeftPropagatedValue;
    type PredictionStore = Connect4PStore;

    fn take_action(&self, game_state: &Self::State, action: &Self::Action) -> Self::State {
        self.engine.take_action(game_state, action)
    }

    fn symmetries(&self, metric: Connect4PositionMetrics) -> Vec<Connect4PositionMetrics> {
        self.mapper.symmetries(metric)
    }

    fn input_size(&self) -> usize {
        INPUT_SIZE
    }

    fn outputs(&self) -> Vec<(String, usize)> {
        vec![
            ("policy".to_string(), OUTPUT_SIZE),
            ("value".to_string(), 1),
            ("moves_left".to_string(), MOVES_LEFT_SIZE),
        ]
    }
}

impl InputMap for Connect4Sampler {
    type State = GameState;

    fn game_state_to_input(&self, game_state: &GameState, input: &mut [f16], mode: Mode) {
        self.mapper.game_state_to_input(game_state, input, mode)
    }
}

impl PredictionsMap for Connect4Sampler {
    type State = GameState;
    type Action = Action;
    type Predictions = Predictions;
    type PropagatedValues = MovesLeftPropagatedValue;

    fn to_output(
        &self,
        game_state: &Self::State,
        targets: Self::Predictions,
        node_metrics: &Connect4NodeMetrics,
    ) -> OutputMap {
        self.mapper.to_output(game_state, targets, node_metrics)
    }
}

impl QMix for Connect4Sampler {
    type State = GameState;
    type Predictions = Predictions;
    type PropagatedValues = MovesLeftPropagatedValue;

    fn mix_q(
        game_state: &Self::State,
        post_blunder_prediction: &Self::Predictions,
        pre_blunder_propagated_values: &Self::PropagatedValues,
        q_mix: f32,
    ) -> Self::Predictions {
        if q_mix == 0.0 {
            return post_blunder_prediction.clone();
        }

        let pre_blunder_value = pre_blunder_propagated_values.value();
        let pre_blunder_game_length = pre_blunder_propagated_values.game_length();

        let player_to_move = game_state.player_to_move();
        let post_blunder_value = post_blunder_prediction.get_value_for_player(player_to_move);
        let post_blunder_game_length = post_blunder_prediction.game_length();

        assert!(
            (0.0..=1.0).contains(&pre_blunder_value) && (0.0..=1.0).contains(&post_blunder_value),
            "blunder_value must be between 0.0 and 1.0"
        );

        assert!(
            post_blunder_game_length >= 0.0 && pre_blunder_game_length >= 0.0,
            "blunder_game_length must be gte 0"
        );

        let mixed_value = ((1.0 - q_mix) * post_blunder_value) + (q_mix * pre_blunder_value);
        let mixed_game_length =
            (1.0 - q_mix) * post_blunder_game_length + q_mix * pre_blunder_game_length;

        assert!(
            (0.0..=1.0).contains(&q_mix),
            "Q mix must be between 0.0 and 1.0"
        );

        let mut value = post_blunder_prediction.value().clone();
        value.update_players_value(player_to_move, mixed_value);

        Predictions::new(value, mixed_game_length)
    }
}

#[derive(Default)]
pub struct Connect4PStore {
    player_value: [Option<Value>; 2],
    game_length: Option<f32>,
}

impl PredictionStore for Connect4PStore {
    type State = GameState;
    type Predictions = Predictions;

    fn get_p_for_player(&self, game_state: &Self::State) -> Option<Self::Predictions> {
        let player = game_state.player_to_move();
        self.player_value[player - 1].as_ref().map(|value| {
            Predictions::new(
                value.clone(),
                self.game_length
                    .expect("Game length should be set before getting predictions"),
            )
        })
    }

    fn set_p_for_player(&mut self, game_state: &Self::State, prediction: Self::Predictions) {
        let player = game_state.player_to_move();
        self.player_value[player - 1] = Some(prediction.value().clone());

        if self.game_length.is_none() {
            self.game_length = Some(prediction.game_length());
        }
    }
}
