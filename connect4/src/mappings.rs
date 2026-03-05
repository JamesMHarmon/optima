use std::collections::HashMap;
use std::convert::TryInto;

use crate::constants::MOVES_LEFT_SIZE;

use super::{
    Action, GameState, INPUT_C, INPUT_H, INPUT_W, OUTPUT_SIZE, Predictions, TranspositionEntry,
    Value, map_board_to_arr,
};
use crate::symmetries::get_symmetries;
use common::{GameLength, PlayerValue, map_moves_left_to_one_hot, moves_left_expected_value};
use itertools::Itertools;
use model::logits::update_logit_policies_to_softmax;
use model::{ActionWithPolicy, GameStateAnalysis, NodeMetrics, PositionMetrics};
use puct::MovesLeftSnapshot;
use tensorflow_model::{InputMap, Mode, PredictionsMap, TranspositionMap};

use half::f16;

#[derive(Default, Clone)]
pub struct Mapper {}

impl Mapper {
    pub fn new() -> Self {
        Self {}
    }

    pub fn symmetries(
        &self,
        metrics: PositionMetrics<GameState, Action, Predictions, MovesLeftSnapshot>,
    ) -> Vec<PositionMetrics<GameState, Action, Predictions, MovesLeftSnapshot>> {
        get_symmetries(metrics)
    }

    fn metrics_to_policy_output(
        &self,
        _game_state: &GameState,
        node_metrics: &NodeMetrics<Action, Predictions, MovesLeftSnapshot>,
    ) -> Vec<f32> {
        //@TODO: Make invalid actions -1.0
        let total_visits = node_metrics.visits() as f32 - 1.0;
        let result: [f32; 7] = node_metrics.children().iter().fold([0.0; 7], |mut r, m| {
            let column_idx = m.action().column() - 1;
            r[column_idx] = m.visits() as f32 / total_visits;
            r
        });

        result.to_vec()
    }

    fn policy_to_valid_actions(
        &self,
        game_state: &GameState,
        policy_scores: &[f16],
    ) -> Vec<ActionWithPolicy<Action>> {
        let mut valid_actions_with_policies = policy_scores
            .iter()
            .zip(1..)
            .filter(|(_, col)| !game_state.is_column_full(*col))
            .map(|(p, col)| ActionWithPolicy::new(col.into(), *p))
            .collect_vec();

        update_logit_policies_to_softmax(&mut valid_actions_with_policies);

        valid_actions_with_policies
    }

    fn metrics_to_value_output(&self, game_state: &GameState, value: &Value) -> Vec<f32> {
        let player_to_move = if game_state.p1_turn_to_move { 1 } else { 2 };
        let val = value.player_value(player_to_move);
        vec![(val * 2.0) - 1.0]
    }

    fn map_value_output_to_value(&self, game_state: &GameState, value_output: f16) -> Value {
        let curr_val = (f16::to_f32(value_output) + 1.0) / 2.0;
        let opp_val = 1.0 - curr_val;
        if game_state.p1_turn_to_move {
            Value::new(curr_val, opp_val)
        } else {
            Value::new(opp_val, curr_val)
        }
    }
}

impl tensorflow_model::Dimension for Mapper {
    fn dimensions(&self) -> [u64; 3] {
        [INPUT_H as u64, INPUT_W as u64, INPUT_C as u64]
    }
}

impl InputMap for Mapper {
    type State = GameState;

    fn game_state_to_input(&self, game_state: &GameState, input: &mut [f16], _mode: Mode) {
        let mut input_vec: Vec<f16> = Vec::with_capacity(INPUT_H * INPUT_W * INPUT_C);

        let (curr_piece_board, opp_piece_board) = if game_state.p1_turn_to_move {
            (game_state.p1_piece_board, game_state.p2_piece_board)
        } else {
            (game_state.p2_piece_board, game_state.p1_piece_board)
        };

        for (curr, opp) in map_board_to_arr(curr_piece_board)
            .iter()
            .zip(map_board_to_arr(opp_piece_board).iter())
        {
            input_vec.push(half::f16::from_f32(*curr));
            input_vec.push(half::f16::from_f32(*opp));
        }

        input.copy_from_slice(input_vec.as_slice());
    }
}

impl PredictionsMap for Mapper {
    type State = GameState;
    type Action = Action;
    type Predictions = Predictions;
    type Snapshot = MovesLeftSnapshot;

    fn to_output(
        &self,
        game_state: &Self::State,
        targets: Self::Predictions,
        node_metrics: &NodeMetrics<Self::Action, Self::Predictions, Self::Snapshot>,
    ) -> std::collections::HashMap<String, Vec<f32>> {
        let policy_output = self.metrics_to_policy_output(game_state, node_metrics);
        let value_output = self.metrics_to_value_output(game_state, targets.value());

        let number_of_actions = game_state.number_of_actions() as f32;
        let moves_left = (targets.game_length() - number_of_actions + 1.0).max(1.0);
        let moves_left_one_hot = map_moves_left_to_one_hot(moves_left, MOVES_LEFT_SIZE);

        /*
            Validate the outputs to ensure values are in their appropriate ranges.
        */
        let sum_of_policy = policy_output.iter().filter(|&&x| x >= 0.0).sum::<f32>();
        assert!(
            f32::abs(sum_of_policy - 1.0) <= f32::EPSILON * policy_output.len() as f32,
            "Policy output should sum to 1.0 but actual sum is {}",
            sum_of_policy
        );

        for policy in policy_output.iter() {
            assert!(
                (0.0..=1.0).contains(policy) || *policy == -1.0,
                "Policy output should be in range 0.0-1.0 but was {}",
                policy
            );
        }

        for value in value_output.iter() {
            assert!(
                (-1.0..=1.0).contains(value),
                "Value output should be in range -1.0-1.0 but was {}",
                value
            );
        }

        assert_eq!(policy_output.len(), OUTPUT_SIZE);
        assert_eq!(value_output.len(), 1);
        assert_eq!(moves_left_one_hot.len(), MOVES_LEFT_SIZE);

        [
            ("policy_head", policy_output),
            ("value_head", value_output),
            ("moves_left_head", moves_left_one_hot),
        ]
        .into_iter()
        .map(|(key, value)| (key.to_string(), value))
        .collect()
    }
}

impl TranspositionMap for Mapper {
    type State = GameState;
    type Action = Action;
    type Predictions = Predictions;
    type TranspositionEntry = TranspositionEntry;

    fn get_transposition_key(&self, game_state: &GameState) -> u64 {
        game_state.get_transposition_hash()
    }

    fn map_output_to_transposition_entry(
        &self,
        game_state: &GameState,
        outputs: HashMap<String, &[f16]>,
    ) -> TranspositionEntry {
        let policy_scores = *outputs
            .get("policy_head")
            .expect("Policy scores not found in output");

        let value = outputs
            .get("value_head")
            .expect("Value not found in output")[0];

        let moves_left_vals = outputs
            .get("moves_left_head")
            .expect("Moves left not found in output");

        let policy_metrics = policy_scores
            .try_into()
            .expect("Slice does not match length of array");

        let moves_left = moves_left_expected_value(moves_left_vals.iter().map(|x| x.to_f32()));
        let number_of_actions = game_state.number_of_actions() as f32;
        let game_length = (number_of_actions + moves_left - 1.0).max(1.0);
        TranspositionEntry::new(policy_metrics, value, game_length)
    }

    fn map_transposition_entry_to_analysis(
        &self,
        game_state: &GameState,
        transposition_entry: &TranspositionEntry,
    ) -> GameStateAnalysis<Action, Predictions> {
        let predictions = Predictions::new(
            self.map_value_output_to_value(game_state, transposition_entry.value()),
            transposition_entry.game_length(),
        );

        GameStateAnalysis::new(
            self.policy_to_valid_actions(game_state, transposition_entry.policy_metrics()),
            predictions,
        )
    }
}
