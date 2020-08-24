use super::action::Action;
use super::board::map_board_to_arr;
use super::constants::{INPUT_C, INPUT_H, INPUT_W, MOVES_LEFT_SIZE, OUTPUT_SIZE};
use super::engine::Engine;
use super::engine::GameState;
use super::value::Value;
use engine::value::Value as ValueTrait;
use model::analytics::ActionWithPolicy;
use model::analytics::GameStateAnalysis;
use model::logits::update_logit_policies_to_softmax;
use model::model::ModelOptions;
use model::model_info::ModelInfo;
use model::node_metrics::NodeMetrics;
use model::position_metrics::PositionMetrics;
use model::tensorflow::get_latest_model_info::get_latest_model_info;
use model::tensorflow::mode::Mode;
use model::tensorflow::model::{TensorflowModel, TensorflowModelOptions};

use anyhow::Result;
use half::f16;

pub struct TranspositionEntry {
    policy_metrics: [f16; OUTPUT_SIZE],
    moves_left: f32,
    value: f16,
}

#[derive(Default)]
pub struct ModelFactory {}

impl ModelFactory {
    pub fn new() -> Self {
        Self {}
    }
}

#[derive(Default)]
pub struct Mapper {}

impl Mapper {
    fn new() -> Self {
        Self {}
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
                    Some(ActionWithPolicy::new(
                        Action::DropPiece((i + 1) as u64),
                        p.to_f32(),
                    ))
                } else {
                    None
                }
            })
            .collect();

        update_logit_policies_to_softmax(&mut valid_actions_with_policies);

        valid_actions_with_policies
    }

    fn map_value_output_to_value(&self, game_state: &GameState, value_output: f32) -> Value {
        let curr_val = (value_output + 1.0) / 2.0;
        let opp_val = 1.0 - curr_val;
        if game_state.p1_turn_to_move {
            Value([curr_val, opp_val])
        } else {
            Value([opp_val, curr_val])
        }
    }
}

impl model::tensorflow::model::Mapper<GameState, Action, Value, TranspositionEntry> for Mapper {
    fn game_state_to_input(&self, game_state: &GameState, _mode: Mode) -> Vec<f16> {
        let mut input: Vec<f32> = Vec::with_capacity(INPUT_H * INPUT_W * INPUT_C);
        let (curr_piece_board, opp_piece_board) = if game_state.p1_turn_to_move {
            (game_state.p1_piece_board, game_state.p2_piece_board)
        } else {
            (game_state.p2_piece_board, game_state.p1_piece_board)
        };

        for (curr, opp) in map_board_to_arr(curr_piece_board)
            .iter()
            .zip(map_board_to_arr(opp_piece_board).iter())
        {
            input.push(*curr);
            input.push(*opp);
        }

        input.into_iter().map(f16::from_f32).collect()
    }

    fn get_input_dimensions(&self) -> [u64; 3] {
        [INPUT_H as u64, INPUT_W as u64, INPUT_C as u64]
    }

    fn get_symmetries(
        &self,
        metrics: PositionMetrics<GameState, Action, Value>,
    ) -> Vec<PositionMetrics<GameState, Action, Value>> {
        //@TODO: Add symmetries.
        vec![metrics]
    }

    fn policy_metrics_to_expected_output(
        &self,
        _game_state: &GameState,
        policy_metrics: &NodeMetrics<Action>,
    ) -> Vec<f32> {
        let total_visits = policy_metrics.visits as f32 - 1.0;
        let result: [f32; 7] =
            policy_metrics
                .children
                .iter()
                .fold([0.0; 7], |mut r, (action, _w, visits)| {
                    match *action {
                        Action::DropPiece(column) => {
                            r[column as usize - 1] = *visits as f32 / total_visits
                        }
                    };
                    r
                });

        result.to_vec()
    }

    fn map_value_to_value_output(&self, game_state: &GameState, value: &Value) -> f32 {
        let player_to_move = if game_state.p1_turn_to_move { 1 } else { 2 };
        let val = value.get_value_for_player(player_to_move);
        (val * 2.0) - 1.0
    }

    fn get_transposition_key(&self, game_state: &GameState) -> u64 {
        game_state.get_transposition_hash()
    }

    fn map_output_to_transposition_entry<I: Iterator<Item = f16>>(
        &self,
        _game_state: &GameState,
        policy_scores: I,
        value: f16,
        moves_left: f32,
    ) -> TranspositionEntry {
        let mut policy_metrics = [f16::ZERO; OUTPUT_SIZE];

        for (i, score) in policy_scores.enumerate() {
            policy_metrics[i] = score;
        }

        TranspositionEntry {
            policy_metrics,
            moves_left,
            value,
        }
    }

    fn map_transposition_entry_to_analysis(
        &self,
        game_state: &GameState,
        transposition_entry: &TranspositionEntry,
    ) -> GameStateAnalysis<Action, Value> {
        GameStateAnalysis::new(
            self.map_value_output_to_value(game_state, transposition_entry.value.to_f32()),
            self.policy_to_valid_actions(game_state, &transposition_entry.policy_metrics),
            transposition_entry.moves_left,
        )
    }
}

impl model::model::ModelFactory for ModelFactory {
    type M = TensorflowModel<GameState, Action, Value, Engine, Mapper, TranspositionEntry>;
    type O = ModelOptions;

    fn create(&self, model_info: &ModelInfo, options: &Self::O) -> Self::M {
        TensorflowModel::<GameState, Action, Value, Engine, Mapper, TranspositionEntry>::create(
            model_info,
            &TensorflowModelOptions {
                num_filters: options.number_of_filters,
                num_blocks: options.number_of_residual_blocks,
                channel_height: INPUT_H,
                channel_width: INPUT_W,
                channels: INPUT_C,
                output_size: OUTPUT_SIZE,
                moves_left_size: MOVES_LEFT_SIZE,
            },
        )
        .unwrap();

        self.get(model_info)
    }

    fn get(&self, model_info: &ModelInfo) -> Self::M {
        let mapper = Mapper::new();

        TensorflowModel::new(model_info.clone(), Engine::new(), mapper, 4000)
    }

    fn get_latest(&self, model_info: &ModelInfo) -> Result<ModelInfo> {
        Ok(get_latest_model_info(model_info)?)
    }
}
