use std::convert::TryInto;
use std::path::PathBuf;

use super::action::Action;
use super::board::map_board_to_arr;
use super::constants::{INPUT_C, INPUT_H, INPUT_W, OUTPUT_SIZE};
use super::engine::Engine;
use super::engine::GameState;
use super::value::Value;
use common::get_env_usize;
use engine::Value as ValueTrait;
use model::logits::update_logit_policies_to_softmax;
use model::{ActionWithPolicy, BasicGameStateAnalysis, Latest, Load, NodeMetrics, PositionMetrics};
use tensorflow_model::{latest, unarchive, Archive as ArchiveModel};
use tensorflow_model::{InputMap, Mode, PolicyMap, TensorflowModel, TranspositionMap, ValueMap};

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
        metrics: PositionMetrics<GameState, Action, Value>,
    ) -> Vec<PositionMetrics<GameState, Action, Value>> {
        //@TODO: Add symmetries.
        vec![metrics]
    }
}

impl tensorflow_model::Dimension for Mapper {
    fn dimensions(&self) -> [u64; 3] {
        [INPUT_H as u64, INPUT_W as u64, INPUT_C as u64]
    }
}

impl InputMap<GameState> for Mapper {
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

impl PolicyMap<GameState, Action, Value> for Mapper {
    fn policy_metrics_to_expected_output(
        &self,
        _game_state: &GameState,
        policy_metrics: &NodeMetrics<Action, Value>,
    ) -> Vec<f32> {
        let total_visits = policy_metrics.visits as f32 - 1.0;
        let result: [f32; 7] = policy_metrics.children.iter().fold([0.0; 7], |mut r, m| {
            match m.action() {
                Action::DropPiece(column) => {
                    r[*column as usize - 1] = m.visits() as f32 / total_visits
                }
            };
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
}

impl ValueMap<GameState, Value> for Mapper {
    fn map_value_to_value_output(&self, game_state: &GameState, value: &Value) -> f32 {
        let player_to_move = if game_state.p1_turn_to_move { 1 } else { 2 };
        let val = value.get_value_for_player(player_to_move);
        (val * 2.0) - 1.0
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

type TranspositionEntry =
    tensorflow_model::transposition_entry::TranspositionEntry<[f16; OUTPUT_SIZE]>;

impl TranspositionMap<GameState, Action, Value, TranspositionEntry> for Mapper {
    fn get_transposition_key(&self, game_state: &GameState) -> u64 {
        game_state.get_transposition_hash()
    }

    fn map_output_to_transposition_entry(
        &self,
        _game_state: &GameState,
        policy_scores: &[f16],
        value: f16,
        moves_left: f32,
    ) -> TranspositionEntry {
        let policy_metrics = policy_scores
            .try_into()
            .expect("Slice does not match length of array");

        TranspositionEntry::new(policy_metrics, value, moves_left)
    }

    fn map_transposition_entry_to_analysis(
        &self,
        game_state: &GameState,
        transposition_entry: &TranspositionEntry,
    ) -> BasicGameStateAnalysis<Action, Value> {
        BasicGameStateAnalysis::new(
            self.map_value_output_to_value(game_state, transposition_entry.value().to_f32()),
            self.policy_to_valid_actions(game_state, transposition_entry.policy_metrics()),
            transposition_entry.moves_left(),
        )
    }
}

impl Load for ModelFactory {
    type MR = ModelRef;
    type M =
        ArchiveModel<TensorflowModel<GameState, Action, Value, Engine, Mapper, TranspositionEntry>>;

    fn load(&self, model_ref: &Self::MR) -> Result<Self::M> {
        let table_size = get_env_usize("TABLE_SIZE").unwrap_or(0);

        let (model_temp_dir, model_options, model_info) = unarchive(&model_ref.0)?;

        let mapper = Mapper::new();

        let model = TensorflowModel::load(
            model_temp_dir.path().to_path_buf(),
            model_options,
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
