use common::{PropagatedGameLength, PropagatedValue};
use engine::{GameEngine, Value as ValueTrait};
use half::f16;
use model::NodeMetrics;
use quoridor::{
    Action, GameState, INPUT_SIZE, MOVES_LEFT_SIZE, Mapper, OUTPUT_SIZE, Predictions,
    QuoridorPropagatedValue, Value,
};
use tensorflow_model::{Dimension, InputMap, Mode, PredictionsMap};

use crate::q_mix::{PredictionStore, QMix};

use super::sample::Sample;

pub struct QuoridorSampler {
    engine: quoridor::Engine,
    mapper: Mapper,
}

impl QuoridorSampler {
    pub fn new(_mode: Option<String>) -> Self {
        Self {
            engine: quoridor::Engine::new(),
            mapper: Mapper::new(),
        }
    }
}

impl Dimension for QuoridorSampler {
    fn dimensions(&self) -> [u64; 3] {
        self.mapper.dimensions()
    }
}

impl Sample for QuoridorSampler {
    type State = GameState;
    type Action = Action;
    type Predictions = Predictions;
    type PropagatedValues = QuoridorPropagatedValue;
    type PredictionStore = QuoridorVStore;

    fn take_action(
        &self,
        game_state: &<Self as Sample>::State,
        action: &<Self as Sample>::Action,
    ) -> <Self as Sample>::State {
        self.engine.take_action(game_state, action)
    }

    fn symmetries(
        &self,
        metric: model::PositionMetrics<
            <Self as Sample>::State,
            <Self as Sample>::Action,
            <Self as Sample>::Predictions,
            <Self as Sample>::PropagatedValues,
        >,
    ) -> Vec<
        model::PositionMetrics<
            <Self as Sample>::State,
            <Self as Sample>::Action,
            <Self as Sample>::Predictions,
            <Self as Sample>::PropagatedValues,
        >,
    > {
        quoridor::get_symmetries(metric)
    }

    fn input_size(&self) -> usize {
        INPUT_SIZE
    }

    fn outputs(&self) -> Vec<(String, usize)> {
        vec![
            ("policy".to_string(), OUTPUT_SIZE),
            ("value".to_string(), 1),
            ("victory_margin".to_string(), 1),
            ("moves_left".to_string(), MOVES_LEFT_SIZE),
        ]
    }
}

impl InputMap for QuoridorSampler {
    type State = GameState;

    fn game_state_to_input(&self, game_state: &GameState, input: &mut [f16], mode: Mode) {
        self.mapper.game_state_to_input(game_state, input, mode)
    }
}

impl PredictionsMap for QuoridorSampler {
    type State = GameState;
    type Action = Action;
    type Predictions = Predictions;
    type PropagatedValues = QuoridorPropagatedValue;

    fn to_output(
        &self,
        game_state: &Self::State,
        targets: Self::Predictions,
        node_metrics: &NodeMetrics<Self::Action, Self::Predictions, Self::PropagatedValues>,
    ) -> std::collections::HashMap<String, Vec<f32>> {
        self.mapper.to_output(game_state, targets, node_metrics)
    }
}

impl QMix for QuoridorSampler {
    type State = GameState;
    type Predictions = Predictions;
    type PropagatedValues = QuoridorPropagatedValue;

    fn mix_q(
        game_state: &Self::State,
        post_blunder_prediction: &Self::Predictions,
        pre_blunder_propagated_values: &Self::PropagatedValues,
        q_mix: f32,
    ) -> Self::Predictions {
        if q_mix == 0.0 {
            return post_blunder_prediction.clone();
        }

        assert!(
            (0.0..=1.0).contains(&q_mix),
            "Q mix must be between 0.0 and 1.0"
        );

        let pre_blunder_value = pre_blunder_propagated_values.value();
        let pre_blunder_victory_margin = pre_blunder_propagated_values.victory_margin();
        let pre_blunder_game_length = pre_blunder_propagated_values.game_length();

        let player_to_move = game_state.player_to_move();
        let post_blunder_value = post_blunder_prediction.get_value_for_player(player_to_move);
        let post_blunder_victory_margin = post_blunder_prediction.victory_margin();
        let post_blunder_game_length = post_blunder_prediction.game_length();

        let mixed_value = ((1.0 - q_mix) * post_blunder_value) + (q_mix * pre_blunder_value);
        let mut new_mixed_value = post_blunder_prediction.value().clone();
        new_mixed_value.update_players_value(player_to_move, mixed_value);

        assert!(
            (0.0..=1.0).contains(&pre_blunder_value) && (0.0..=1.0).contains(&post_blunder_value),
            "blunder_value must be between 0.0 and 1.0"
        );

        let mixed_victory_margin =
            ((1.0 - q_mix) * post_blunder_victory_margin) + (q_mix * pre_blunder_victory_margin);

        assert!(
            post_blunder_game_length >= 0.0 && pre_blunder_game_length >= 0.0,
            "blunder_game_length must be gte 0"
        );

        let mixed_game_length =
            (1.0 - q_mix) * post_blunder_game_length + q_mix * pre_blunder_game_length;

        Predictions::new(new_mixed_value, mixed_victory_margin, mixed_game_length)
    }
}

#[derive(Default)]
pub struct QuoridorVStore {
    player_value: [Option<Value>; 2],
    victory_margin: Option<f32>,
    game_length: Option<f32>,
}

#[allow(non_snake_case)]
impl PredictionStore for QuoridorVStore {
    type State = GameState;
    type Predictions = Predictions;

    fn get_p_for_player(&self, game_state: &Self::State) -> Option<Self::Predictions> {
        let player = game_state.player_to_move();
        self.player_value[player - 1].as_ref().map(|value| {
            Predictions::new(
                value.clone(),
                self.victory_margin
                    .expect("Victory margin should be set before getting predictions"),
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

        if self.victory_margin.is_none() {
            self.victory_margin = Some(prediction.victory_margin());
        }
    }
}
