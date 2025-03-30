use common::{MovesLeftPropagatedValue, PropagatedGameLength, PropagatedValue};
use engine::{GameEngine, Value as ValueTrait};
use half::f16;
use model::NodeMetrics;
use quoridor::{Action, GameState, Mapper, Predictions, INPUT_SIZE, MOVES_LEFT_SIZE, OUTPUT_SIZE};
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
    type PropagatedValues = MovesLeftPropagatedValue;
    type PredictionStore = QuoridorVStore;

    fn take_action(
        &self,
        game_state: &<Self as Sample>::State,
        action: &<Self as Sample>::Action,
    ) -> <Self as Sample>::State {
        self.engine.take_action(game_state, action)
    }

    fn move_number(&self, game_state: &<Self as Sample>::State) -> usize {
        self.engine.move_number(game_state)
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
    type PropagatedValues = MovesLeftPropagatedValue;

    fn to_output(
        &self,
        game_state: &Self::State,
        node_metrics: &NodeMetrics<Self::Action, Self::Predictions, Self::PropagatedValues>,
    ) -> std::collections::HashMap<String, Vec<f32>> {
        self.mapper.to_output(game_state, node_metrics)
    }
}

impl QMix for QuoridorSampler {
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

        // @TODO: Fix this value and game_length. This may need to be divided by Nsa.
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
        let predictions = Predictions::new(value, mixed_game_length);

        predictions
    }
}

#[derive(Default)]
pub struct QuoridorVStore([Option<Predictions>; 2]);

#[allow(non_snake_case)]
impl PredictionStore for QuoridorVStore {
    type State = GameState;
    type Predictions = Predictions;

    fn get_p_for_player(&self, game_state: &Self::State) -> Option<&Self::Predictions> {
        let player = game_state.player_to_move();
        self.0[player - 1].as_ref()
    }

    fn set_p_for_player(&mut self, game_state: &Self::State, prediction: Self::Predictions) {
        let player = game_state.player_to_move();
        self.0[player - 1] = Some(prediction);
    }
}
