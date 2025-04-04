use std::collections::HashMap;
use std::convert::TryInto;
use std::path::PathBuf;

use crate::constants::MOVES_LEFT_SIZE;

use super::{
    map_board_to_arr, Action, Engine, GameState, Predictions, Value, INPUT_C, INPUT_H, INPUT_W,
    OUTPUT_SIZE,
};
use common::{get_env_usize, MovesLeftPropagatedValue};
use engine::Value as ValueTrait;
use mcts::map_moves_left_to_one_hot;
use model::logits::update_logit_policies_to_softmax;
use model::{ActionWithPolicy, GameStateAnalysis, Latest, Load, NodeMetrics, PositionMetrics};
use tensorflow_model::{latest, unarchive, Archive as ArchiveModel};
use tensorflow_model::{InputMap, Mode, PredictionsMap, TensorflowModel, TranspositionMap};

use anyhow::Result;
use half::f16;

#[derive(Default)]
pub struct ModelFactory {
    model_dir: PathBuf,
}

impl ModelFactory {
    pub fn new(model_dir: PathBuf) -> Self {
        ModelFactory { model_dir }
    }
}

#[derive(Default)]
pub struct Mapper {}

impl Mapper {
    fn new() -> Self {
        Self {}
    }

    pub fn symmetries(
        &self,
        metrics: PositionMetrics<GameState, Action, Predictions, MovesLeftPropagatedValue>,
    ) -> Vec<PositionMetrics<GameState, Action, Predictions, MovesLeftPropagatedValue>> {
        //@TODO: Add symmetries.
        vec![metrics]
    }

    fn metrics_to_policy_output(
        &self,
        _game_state: &GameState,
        node_metrics: &NodeMetrics<Action, Predictions, MovesLeftPropagatedValue>,
    ) -> Vec<f32> {
        //@TODO: Make invalid actions -1.0
        let total_visits = node_metrics.visits as f32 - 1.0;
        let result: [f32; 7] = node_metrics.children.iter().fold([0.0; 7], |mut r, m| {
            let column_idx = m.action().column() as usize - 1;
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
        let mut valid_actions_with_policies: Vec<ActionWithPolicy<Action>> = game_state
            .get_valid_actions()
            .iter()
            .zip(policy_scores)
            .enumerate()
            .filter_map(|(i, (v, p))| {
                if *v {
                    Some(ActionWithPolicy::new(Action::DropPiece((i + 1) as u8), *p))
                } else {
                    None
                }
            })
            .collect();

        update_logit_policies_to_softmax(&mut valid_actions_with_policies);

        valid_actions_with_policies
    }

    fn metrics_to_value_output(&self, game_state: &GameState, value: &Value) -> Vec<f32> {
        let player_to_move = if game_state.p1_turn_to_move { 1 } else { 2 };
        let val = value.get_value_for_player(player_to_move);
        vec![(val * 2.0) - 1.0]
    }

    fn map_value_output_to_value(&self, game_state: &GameState, value_output: f16) -> Value {
        let curr_val = (f16::to_f32(value_output) + 1.0) / 2.0;
        let opp_val = 1.0 - curr_val;
        if game_state.p1_turn_to_move {
            Value([curr_val, opp_val])
        } else {
            Value([opp_val, curr_val])
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
    type PropagatedValues = MovesLeftPropagatedValue;

    fn to_output(
        &self,
        game_state: &Self::State,
        targets: Self::Predictions,
        node_metrics: &NodeMetrics<Self::Action, Self::Predictions, Self::PropagatedValues>,
    ) -> std::collections::HashMap<String, Vec<f32>> {
        let policy_output = self.metrics_to_policy_output(game_state, node_metrics);
        let value_output = self.metrics_to_value_output(game_state, targets.value());

        // @TODO: Verify how number_of_actions relates to game_length.
        let move_number = game_state.number_of_actions() as f32;
        let moves_left = (targets.game_length() - move_number).max(0.0);
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
            ("policy", policy_output),
            ("value", value_output),
            ("moves_left", moves_left_one_hot),
        ]
        .into_iter()
        .map(|(key, value)| (key.to_string(), value))
        .collect()
    }
}

pub struct TranspositionEntry {
    policy_metrics: [f16; OUTPUT_SIZE],
    value: f16,
    game_length: f32,
}

impl TranspositionEntry {
    fn new(policy_metrics: [f16; OUTPUT_SIZE], value: f16, game_length: f32) -> Self {
        Self {
            policy_metrics,
            value,
            game_length,
        }
    }

    fn policy_metrics(&self) -> &[f16; OUTPUT_SIZE] {
        &self.policy_metrics
    }

    fn value(&self) -> f16 {
        self.value
    }

    fn game_length(&self) -> f32 {
        self.game_length
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
            .get("policy")
            .expect("Policy scores not found in output");

        let value = outputs.get("value").expect("Value not found in output")[0];

        let moves_left = outputs
            .get("moves_left")
            .expect("Moves left not found in output")[0];

        let policy_metrics = policy_scores
            .try_into()
            .expect("Slice does not match length of array");

        let game_length = game_state.number_of_actions() as f32 + f16::to_f32(moves_left);
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

impl Load for ModelFactory {
    type MR = ModelRef;
    type M = ArchiveModel<
        TensorflowModel<GameState, Action, Predictions, Engine, Mapper, TranspositionEntry>,
    >;

    fn load(&self, model_ref: &Self::MR) -> Result<Self::M> {
        let table_size = get_env_usize("TABLE_SIZE").unwrap_or(0);

        let (model_temp_dir, _, model_info) = unarchive(&model_ref.0)?;

        let mapper = Mapper::new();

        let model = TensorflowModel::load(
            model_temp_dir.path().to_path_buf(),
            model_info,
            Engine::new(),
            mapper,
            table_size,
        )?;

        Ok(ArchiveModel::new(model, model_temp_dir))
    }
}

#[derive(Debug, Eq, PartialEq)]
pub struct ModelRef(PathBuf);

impl Latest for ModelFactory {
    type MR = ModelRef;

    fn latest(&self) -> Result<Self::MR> {
        latest(&self.model_dir).map(ModelRef)
    }
}
