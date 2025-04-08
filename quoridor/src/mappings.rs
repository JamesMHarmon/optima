use crate::{Predictions, QuoridorPropagatedValue, BOARD_SIZE, MOVES_LEFT_SIZE};

use super::transposition_entry::TranspositionEntry;
use half::f16;
use mcts::{map_moves_left_to_one_hot, moves_left_expected_value};
use std::collections::HashMap;
use std::convert::TryInto;
use std::vec;
use tensorflow_model::{InputMap, PredictionsMap, TranspositionMap};

use super::constants::{
    ASCII_LETTER_A, BOARD_HEIGHT, BOARD_WIDTH, INPUT_C, INPUT_H, INPUT_W, NUM_WALLS_PER_PLAYER,
    OUTPUT_SIZE, PAWN_BOARD_SIZE, WALL_BOARD_SIZE,
};
use super::{Action, ActionType, Coordinate, GameState, Value};
use engine::Value as ValueTrait;
use model::logits::update_logit_policies_to_softmax;
use model::{ActionWithPolicy, ConvInputBuilder, GameStateAnalysis, NodeMetrics};
use tensorflow_model::Mode;

#[derive(Clone, Default)]
pub struct Mapper {}

impl Mapper {
    pub fn new() -> Self {
        Self {}
    }

    fn metrics_to_policy_output(
        &self,
        game_state: &GameState,
        policy_metrics: &NodeMetrics<Action, Predictions, QuoridorPropagatedValue>,
    ) -> Vec<f32> {
        let total_visits = policy_metrics.visits as f32 - 1.0;
        let rotate: bool = !game_state.p1_turn_to_move();

        let inputs = vec![-1f32; OUTPUT_SIZE];

        policy_metrics.children.iter().fold(inputs, |mut r, m| {
            // Policy scores for quoridor should be in the perspective of player 1. That means that if we are p2, we need to flip the actions as if we were looking
            // at the board from the perspective of player 1, but with the pieces rotated.
            let input_idx = if rotate {
                map_action_to_output_idx(&m.action().rotate())
            } else {
                map_action_to_output_idx(m.action())
            };

            r[input_idx] = m.visits() as f32 / total_visits;
            r
        })
    }

    fn policy_to_valid_actions(
        &self,
        game_state: &GameState,
        policy_scores: &[f16],
    ) -> Vec<ActionWithPolicy<Action>> {
        let valid_actions = game_state.valid_actions();

        let rotate = !game_state.p1_turn_to_move();
        let mut valid_actions_with_policies: Vec<ActionWithPolicy<Action>> = valid_actions
            .map(|a| {
                // Policy scores coming from the quoridor model are always from the perspective of player 1.
                // This means that if we are p2, we need to flip the actions coming back and translate them
                // to be actions in the p2 perspective.
                let p_idx = if rotate {
                    map_action_to_output_idx(&a.rotate())
                } else {
                    map_action_to_output_idx(&a)
                };

                let p = policy_scores[p_idx];

                ActionWithPolicy::new(a, p)
            })
            .collect();

        update_logit_policies_to_softmax(&mut valid_actions_with_policies);

        valid_actions_with_policies
    }

    fn metrics_to_value_output(&self, game_state: &GameState, value: &Value) -> Vec<f32> {
        let player_to_move = game_state.player_to_move();
        let val = value.get_value_for_player(player_to_move);
        vec![(val * 2.0) - 1.0]
    }

    fn map_value_output_to_value(&self, game_state: &GameState, value_output: f16) -> Value {
        let curr_val = (f16::to_f32(value_output) + 1.0) / 2.0;
        let opp_val = 1.0 - curr_val;
        if game_state.p1_turn_to_move() {
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
        let mut builder = ConvInputBuilder::new(BOARD_SIZE, input);

        let rotate: bool = !game_state.p1_turn_to_move();
        let rotate_pawn = |c: Coordinate| if rotate { c.rotate(false) } else { c };
        let rotate_wall = |c: Coordinate| if rotate { c.rotate(true) } else { c };

        let curr_player_info = game_state.curr_player();
        let opp_player_info = game_state.opp_player();

        let curr_pawn = rotate_pawn(curr_player_info.pawn());
        let opp_pawn = rotate_pawn(opp_player_info.pawn());
        let vertical_walls = game_state
            .vertical_walls()
            .map(rotate_wall)
            .map(|c| c.index());
        let horizontal_walls = game_state
            .horizontal_walls()
            .map(rotate_wall)
            .map(|c| c.index());

        let curr_walls_norm = (curr_player_info.num_walls() as f32) / NUM_WALLS_PER_PLAYER as f32;
        let curr_walls_norm = f16::from_f32(curr_walls_norm);
        let opp_walls_norm = (opp_player_info.num_walls() as f32) / NUM_WALLS_PER_PLAYER as f32;
        let opp_walls_norm = f16::from_f32(opp_walls_norm);

        builder.channel(0).write_at_idx(curr_pawn.index(), f16::ONE);
        builder.channel(1).write_at_idx(opp_pawn.index(), f16::ONE);
        builder.channel(2).set_bits_at_indexes(vertical_walls);
        builder.channel(3).set_bits_at_indexes(horizontal_walls);
        builder.channel(4).fill(curr_walls_norm);
        builder.channel(5).fill(opp_walls_norm);
    }
}

impl PredictionsMap for Mapper {
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
        let policy_output = self.metrics_to_policy_output(game_state, node_metrics);
        let value_output = self.metrics_to_value_output(game_state, targets.value());
        let victory_margin_output = vec![targets.victory_margin()];

        let move_number = game_state.move_number() as f32;
        let moves_left = (targets.game_length() - move_number + 1.0).max(1.0);
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

        for victory_margin in victory_margin_output.iter() {
            assert!(
                *victory_margin >= 0.0 && *victory_margin <= BOARD_SIZE as f32 * BOARD_SIZE as f32,
                "Victory Margin output should be >= 0.0 but was {}",
                victory_margin
            );
        }

        assert_eq!(policy_output.len(), OUTPUT_SIZE);
        assert_eq!(value_output.len(), 1);
        assert_eq!(victory_margin_output.len(), 1);
        assert_eq!(moves_left_one_hot.len(), MOVES_LEFT_SIZE);

        [
            ("policy", policy_output),
            ("value", value_output),
            ("victory_margin", victory_margin_output),
            ("moves_left", moves_left_one_hot),
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
        game_state.transposition_hash()
    }

    fn map_output_to_transposition_entry(
        &self,
        game_state: &GameState,
        outputs: HashMap<String, &[f16]>,
    ) -> TranspositionEntry {
        let policy_scores = *outputs
            .get("policy_head")
            .expect("Policy scores not found in output");

        let policy_metrics = policy_scores
            .try_into()
            .expect("Slice does not match length of array");

        let value = outputs
            .get("value_head")
            .expect("Value not found in output")[0];

        let victory_margin = outputs
            .get("victory_margin_head")
            .expect("Victory margin not found in output")[0];

        let moves_left_vals = outputs
            .get("moves_left_head")
            .expect("Moves left not found in output");

        let moves_left = moves_left_expected_value(moves_left_vals.iter().map(|x| x.to_f32()));

        let game_length = (game_state.move_number() as f32 + moves_left - 1.0).max(1.0);

        TranspositionEntry::new(policy_metrics, value, victory_margin, game_length)
    }

    fn map_transposition_entry_to_analysis(
        &self,
        game_state: &GameState,
        transposition_entry: &TranspositionEntry,
    ) -> GameStateAnalysis<Action, Predictions> {
        let predictions = Predictions::new(
            self.map_value_output_to_value(game_state, transposition_entry.value()),
            f16::to_f32(transposition_entry.victory_margin()),
            transposition_entry.game_length(),
        );

        GameStateAnalysis::new(
            self.policy_to_valid_actions(game_state, transposition_entry.policy_metrics()),
            predictions,
        )
    }
}

fn map_action_to_output_idx(action: &Action) -> usize {
    let len_moves_inputs = PAWN_BOARD_SIZE;
    let len_wall_inputs = WALL_BOARD_SIZE;

    match action.action_type() {
        ActionType::PawnMove => map_coord_to_output_idx_nine_by_nine(&action.coord()),
        ActionType::VerticalWall => {
            map_coord_to_output_idx_eight_by_eight(&action.coord()) + len_moves_inputs
        }
        ActionType::HorizontalWall => {
            map_coord_to_output_idx_eight_by_eight(&action.coord()) + len_moves_inputs + len_wall_inputs
        }
        ActionType::Pass => len_moves_inputs + len_wall_inputs * 2,
    }
}

fn map_coord_to_output_idx_nine_by_nine(coord: &Coordinate) -> usize {
    let col_idx = (coord.col() as u8 - ASCII_LETTER_A) as usize;

    col_idx + ((BOARD_HEIGHT - coord.row()) * BOARD_WIDTH)
}

fn map_coord_to_output_idx_eight_by_eight(coord: &Coordinate) -> usize {
    let col_idx = (coord.col() as u8 - ASCII_LETTER_A) as usize;

    col_idx + ((BOARD_HEIGHT - coord.row()) * (BOARD_WIDTH - 1))
}

#[cfg(test)]
mod tests {
    use std::ops::RangeInclusive;

    use super::*;
    use assert_approx_eq::assert_approx_eq;
    use engine::game_state::GameState as GameStateTrait;
    use itertools::Itertools;

    fn map_to_input_vec(
        curr_pawn_idx: usize,
        opp_pawn_idx: usize,
        vertical_wall_idxs: &[usize],
        horizontal_wall_idxs: &[usize],
        curr_num_walls: usize,
        opp_num_walls: usize,
    ) -> Vec<f32> {
        let offset = INPUT_C;
        let mut output = [0.0; INPUT_W * INPUT_H * INPUT_C];

        output[curr_pawn_idx * offset] = 1.0;
        output[opp_pawn_idx * offset + 1] = 1.0;

        for idx in vertical_wall_idxs {
            output[idx * offset + 2] = 1.0;
        }

        for idx in horizontal_wall_idxs {
            output[idx * offset + 3] = 1.0;
        }

        for idx in 0..81 {
            output[idx * offset + 4] = curr_num_walls as f32 / 10.0;
        }

        for idx in 0..81 {
            output[idx * offset + 5] = opp_num_walls as f32 / 10.0;
        }

        output.to_vec()
    }

    fn assert_approximate_eq_slice(a: &[f32], b: &[f32]) {
        assert_eq!(a.len(), b.len());
        for (a, b) in a.iter().zip(b) {
            assert_approx_eq!(*a, *b, f16::EPSILON.to_f32());
        }
    }

    fn game_state_to_input(game_state: &GameState) -> Vec<f32> {
        let mut input = [f16::ZERO; INPUT_H * INPUT_W * INPUT_C];

        let mapper = Mapper::new();
        mapper.game_state_to_input(game_state, &mut input, Mode::Infer);

        input.iter().copied().map(f16::to_f32).collect::<Vec<_>>()
    }

    #[test]
    fn test_map_coord_to_output_idx_nine_by_nine_a1() {
        let coord = "a1".parse::<Coordinate>().unwrap();
        let idx = map_coord_to_output_idx_nine_by_nine(&coord);

        assert_eq!(72, idx);
    }

    #[test]
    fn test_map_coord_to_output_idx_nine_by_nine_a9() {
        let coord = "a9".parse::<Coordinate>().unwrap();
        let idx = map_coord_to_output_idx_nine_by_nine(&coord);

        assert_eq!(0, idx);
    }

    #[test]
    fn test_map_coord_to_output_idx_nine_by_nine_i1() {
        let coord = "i1".parse::<Coordinate>().unwrap();
        let idx = map_coord_to_output_idx_nine_by_nine(&coord);

        assert_eq!(80, idx);
    }

    #[test]
    fn test_map_coord_to_output_idx_nine_by_nine_i9() {
        let coord = "i9".parse::<Coordinate>().unwrap();
        let idx = map_coord_to_output_idx_nine_by_nine(&coord);

        assert_eq!(8, idx);
    }

    #[test]
    fn test_map_coord_to_output_idx_nine_by_nine_e5() {
        let coord = "e5".parse::<Coordinate>().unwrap();
        let idx = map_coord_to_output_idx_nine_by_nine(&coord);

        assert_eq!(40, idx);
    }

    #[test]
    fn test_map_coord_to_output_idx_eight_by_eight_a2() {
        let coord = "a2".parse::<Coordinate>().unwrap();
        let idx = map_coord_to_output_idx_eight_by_eight(&coord);

        assert_eq!(56, idx);
    }

    #[test]
    fn test_map_coord_to_output_idx_eight_by_eight_a9() {
        let coord = "a9".parse::<Coordinate>().unwrap();
        let idx = map_coord_to_output_idx_eight_by_eight(&coord);

        assert_eq!(0, idx);
    }

    #[test]
    fn test_map_coord_to_output_idx_eight_by_eight_h2() {
        let coord = "h2".parse::<Coordinate>().unwrap();
        let idx = map_coord_to_output_idx_eight_by_eight(&coord);

        assert_eq!(63, idx);
    }

    #[test]
    fn test_map_coord_to_output_idx_eight_by_eight_h9() {
        let coord = "h9".parse::<Coordinate>().unwrap();
        let idx = map_coord_to_output_idx_eight_by_eight(&coord);

        assert_eq!(7, idx);
    }

    #[test]
    fn test_map_coord_to_output_idx_eight_by_eight_e6() {
        let coord = "e6".parse::<Coordinate>().unwrap();
        let idx = map_coord_to_output_idx_eight_by_eight(&coord);

        assert_eq!(28, idx);
    }

    #[test]
    fn test_map_action_to_output_idx_pawn_a9() {
        let action = "a9".parse().unwrap();
        let idx = map_action_to_output_idx(&action);

        assert_eq!(0, idx);
    }

    #[test]
    fn test_map_action_to_output_idx_pawn_i1() {
        let action = "i1".parse().unwrap();
        let idx = map_action_to_output_idx(&action);

        assert_eq!(80, idx);
    }

    #[test]
    fn test_map_action_to_output_idx_vertical_wall_a9() {
        let action = "a9v".parse().unwrap();
        let idx = map_action_to_output_idx(&action);

        assert_eq!(81, idx);
    }

    #[test]
    fn test_map_action_to_output_idx_vertical_wall_h2() {
        let action = "h2v".parse().unwrap();
        let idx = map_action_to_output_idx(&action);

        assert_eq!(144, idx);
    }

    #[test]
    fn test_map_action_to_output_idx_horizontal_wall_a9() {
        let action = "a9h".parse().unwrap();
        let idx = map_action_to_output_idx(&action);

        assert_eq!(145, idx);
    }

    #[test]
    fn test_map_action_to_output_idx_horizontal_wall_h2() {
        let action = "h2h".parse().unwrap();
        let idx = map_action_to_output_idx(&action);

        assert_eq!(208, idx);
    }

    #[test]
    fn test_map_action_to_output_idx_all_actions() {
        let generate_actions =
            |action_type, col_range: RangeInclusive<char>, row_range: RangeInclusive<usize>| {
                row_range.rev().flat_map(move |row| {
                    col_range
                        .clone()
                        .map(move |col| Action::from((action_type, Coordinate::new(col, row))))
                })
            };

        let pawn_actions = || generate_actions(ActionType::PawnMove, 'a'..='i', 1..=9);
        let horizontal_wall_actions =
            || generate_actions(ActionType::VerticalWall, 'a'..='h', 2..=9);
        let vertical_wall_actions =
            || generate_actions(ActionType::HorizontalWall, 'a'..='h', 2..=9);

        let all_actions = pawn_actions()
            .chain(horizontal_wall_actions())
            .chain(vertical_wall_actions())
            .chain(std::iter::once(Action::pass()))
            .collect_vec();

        assert_eq!(all_actions.len(), OUTPUT_SIZE);

        for (idx, action) in all_actions.iter().enumerate() {
            assert_eq!(idx, map_action_to_output_idx(action));
        }
    }

    #[test]
    fn test_game_state_to_input_initial_p1() {
        let game_state = GameState::initial();

        let input = game_state_to_input(&game_state);

        assert_approximate_eq_slice(&map_to_input_vec(76, 4, &[], &[], 10, 10), &input)
    }

    #[test]
    fn test_game_state_to_input_initial_p2() {
        let mut game_state = GameState::initial();
        game_state.take_action(&"e2".parse::<Action>().unwrap());

        let input = game_state_to_input(&game_state);

        assert_approximate_eq_slice(&map_to_input_vec(76, 13, &[], &[], 10, 10), &input)
    }

    #[test]
    fn test_game_state_to_input_walls_remaining() {
        let mut game_state = GameState::initial();
        game_state.take_action(&"h2h".parse::<Action>().unwrap());

        let input = game_state_to_input(&game_state);

        assert_approximate_eq_slice(&map_to_input_vec(76, 4, &[], &[0], 10, 9), &input)
    }

    #[test]
    fn test_game_state_to_input_walls_remaining_p2() {
        let mut game_state = GameState::initial();
        game_state.take_action(&"h2h".parse::<Action>().unwrap());
        game_state.take_action(&"d9".parse::<Action>().unwrap());

        let input = game_state_to_input(&game_state);

        assert_approximate_eq_slice(&map_to_input_vec(76, 3, &[], &[70], 9, 10), &input)
    }

    #[test]
    fn test_game_state_to_input_vertical_walls() {
        let mut game_state = GameState::initial();
        game_state.take_action(&"e2".parse::<Action>().unwrap());
        game_state.take_action(&"c4v".parse::<Action>().unwrap());

        let input = game_state_to_input(&game_state);

        assert_approximate_eq_slice(&map_to_input_vec(67, 4, &[47], &[], 10, 9), &input)
    }

    #[test]
    fn test_game_state_to_input_vertical_walls_p2() {
        let mut game_state = GameState::initial();
        game_state.take_action(&"e2".parse::<Action>().unwrap());
        game_state.take_action(&"c4v".parse::<Action>().unwrap());
        game_state.take_action(&"e3".parse::<Action>().unwrap());

        let input = game_state_to_input(&game_state);

        assert_approximate_eq_slice(&map_to_input_vec(76, 22, &[23], &[], 9, 10), &input)
    }
}
