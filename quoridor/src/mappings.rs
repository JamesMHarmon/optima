use half::f16;
use itertools::izip;
use std::convert::TryInto;
use tensorflow_model::{InputMap, PolicyMap, TranspositionMap, ValueMap};

use crate::ActionExpanded;

use super::transposition_entry::TranspositionEntry;

use super::board::{map_board_to_arr_rotatable, BoardType};
use super::constants::{
    ASCII_LETTER_A, BOARD_HEIGHT, BOARD_WIDTH, INPUT_C, INPUT_H, INPUT_W, NUM_WALLS_PER_PLAYER,
    OUTPUT_SIZE, PAWN_BOARD_SIZE, WALL_BOARD_SIZE,
};
use super::{Action, Coordinate, GameState, Value};
use engine::Value as ValueTrait;
use model::logits::update_logit_policies_to_softmax;
use model::{ActionWithPolicy, BasicGameStateAnalysis, NodeMetrics};
use tensorflow_model::Mode;

#[derive(Default)]
pub struct Mapper {}

impl Mapper {
    pub fn new() -> Self {
        Self {}
    }
}

impl tensorflow_model::Dimension for Mapper {
    fn dimensions(&self) -> [u64; 3] {
        [INPUT_H as u64, INPUT_W as u64, INPUT_C as u64]
    }
}

impl PolicyMap<GameState, Action, Value> for Mapper {
    fn policy_metrics_to_expected_output(
        &self,
        game_state: &GameState,
        policy_metrics: &NodeMetrics<Action, Value>,
    ) -> Vec<f32> {
        let total_visits = policy_metrics.visits as f32 - 1.0;
        let rotate: bool = !game_state.p1_turn_to_move;
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
        let valid_pawn_moves = game_state.get_valid_pawn_move_actions();
        let valid_vert_walls = game_state.get_valid_vertical_wall_actions();
        let valid_horiz_walls = game_state.get_valid_horizontal_wall_actions();
        let actions = valid_pawn_moves
            .chain(valid_vert_walls)
            .chain(valid_horiz_walls);
        let rotate = !game_state.p1_turn_to_move;

        let mut valid_actions_with_policies: Vec<ActionWithPolicy<Action>> = actions
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

impl InputMap<GameState> for Mapper {
    fn game_state_to_input(&self, game_state: &GameState, input: &mut [f16], _mode: Mode) {
        let mut input_vec: Vec<f16> = Vec::with_capacity(INPUT_H * INPUT_W * INPUT_C);

        let GameState {
            p1_turn_to_move,
            p1_pawn_board,
            p2_pawn_board,
            vertical_wall_board: vertical_wall_placement_board,
            horizontal_wall_board: horizontal_wall_placement_board,
            p1_num_walls_placed,
            p2_num_walls_placed,
            ..
        } = game_state;

        let curr_player_pawn_board = if *p1_turn_to_move {
            *p1_pawn_board
        } else {
            *p2_pawn_board
        };
        let opp_player_pawn_board = if *p1_turn_to_move {
            *p2_pawn_board
        } else {
            *p1_pawn_board
        };

        let rotate = !*p1_turn_to_move;

        let curr_pawn_board_vec =
            map_board_to_arr_rotatable(curr_player_pawn_board, BoardType::Pawn, rotate);
        let opp_pawn_board_vec =
            map_board_to_arr_rotatable(opp_player_pawn_board, BoardType::Pawn, rotate);
        let vertical_wall_vec = map_board_to_arr_rotatable(
            *vertical_wall_placement_board,
            BoardType::VerticalWall,
            rotate,
        );
        let horizontal_wall_vec = map_board_to_arr_rotatable(
            *horizontal_wall_placement_board,
            BoardType::HorizontalWall,
            rotate,
        );

        let curr_num_walls_placed = if *p1_turn_to_move {
            p1_num_walls_placed
        } else {
            p2_num_walls_placed
        };
        let opp_num_walls_placed = if *p1_turn_to_move {
            p2_num_walls_placed
        } else {
            p1_num_walls_placed
        };
        let curr_num_walls_placed_norm =
            (*curr_num_walls_placed as f32) / NUM_WALLS_PER_PLAYER as f32;
        let opp_num_walls_placed_norm =
            (*opp_num_walls_placed as f32) / NUM_WALLS_PER_PLAYER as f32;

        for (curr_pawn, opp_pawn, vw, hw) in izip!(
            curr_pawn_board_vec.iter(),
            opp_pawn_board_vec.iter(),
            vertical_wall_vec.iter(),
            horizontal_wall_vec.iter()
        ) {
            input_vec.push(f16::from_f32(*curr_pawn));
            input_vec.push(f16::from_f32(*opp_pawn));
            input_vec.push(f16::from_f32(*vw));
            input_vec.push(f16::from_f32(*hw));
            input_vec.push(f16::from_f32(curr_num_walls_placed_norm));
            input_vec.push(f16::from_f32(opp_num_walls_placed_norm));
        }

        input.copy_from_slice(input_vec.as_slice())
    }
}

impl TranspositionMap<GameState, Action, Value, TranspositionEntry> for Mapper {
    fn get_transposition_key(&self, game_state: &GameState) -> u64 {
        game_state.transposition_hash()
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

fn map_action_to_output_idx(action: &Action) -> usize {
    let len_moves_inputs = PAWN_BOARD_SIZE;
    let len_wall_inputs = WALL_BOARD_SIZE;

    match (*action).into() {
        ActionExpanded::MovePawn(coord) => map_coord_to_input_idx_nine_by_nine(&coord),
        ActionExpanded::PlaceVerticalWall(coord) => {
            map_coord_to_input_idx_eight_by_eight(&coord) + len_moves_inputs
        }
        ActionExpanded::PlaceHorizontalWall(coord) => {
            map_coord_to_input_idx_eight_by_eight(&coord) + len_moves_inputs + len_wall_inputs
        }
    }
}

fn map_coord_to_input_idx_nine_by_nine(coord: &Coordinate) -> usize {
    let col_idx = (coord.col() as u8 - ASCII_LETTER_A) as usize;

    col_idx + ((BOARD_HEIGHT - coord.row()) * BOARD_WIDTH)
}

fn map_coord_to_input_idx_eight_by_eight(coord: &Coordinate) -> usize {
    let col_idx = (coord.col() as u8 - ASCII_LETTER_A) as usize;

    col_idx + ((BOARD_HEIGHT - coord.row() - 1) * (BOARD_WIDTH - 1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_approx_eq::assert_approx_eq;
    use engine::game_state::GameState as GameStateTrait;

    fn map_to_input_vec(
        curr_pawn_idx: usize,
        opp_pawn_idx: usize,
        vertical_wall_idxs: &[usize],
        horizontal_wall_idxs: &[usize],
        curr_num_walls_placed: usize,
        opp_num_walls_placed: usize,
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
            output[idx * offset + 4] = curr_num_walls_placed as f32 / 10.0;
        }

        for idx in 0..81 {
            output[idx * offset + 5] = opp_num_walls_placed as f32 / 10.0;
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
    fn test_map_coord_to_input_idx_nine_by_nine_a1() {
        let coord = "a1".parse::<Coordinate>().unwrap();
        let idx = map_coord_to_input_idx_nine_by_nine(&coord);

        assert_eq!(72, idx);
    }

    #[test]
    fn test_map_coord_to_input_idx_nine_by_nine_a9() {
        let coord = "a9".parse::<Coordinate>().unwrap();
        let idx = map_coord_to_input_idx_nine_by_nine(&coord);

        assert_eq!(0, idx);
    }

    #[test]
    fn test_map_coord_to_input_idx_nine_by_nine_i1() {
        let coord = "i1".parse::<Coordinate>().unwrap();
        let idx = map_coord_to_input_idx_nine_by_nine(&coord);

        assert_eq!(80, idx);
    }

    #[test]
    fn test_map_coord_to_input_idx_nine_by_nine_i9() {
        let coord = "i9".parse::<Coordinate>().unwrap();
        let idx = map_coord_to_input_idx_nine_by_nine(&coord);

        assert_eq!(8, idx);
    }

    #[test]
    fn test_map_coord_to_input_idx_nine_by_nine_e5() {
        let coord = "e5".parse::<Coordinate>().unwrap();
        let idx = map_coord_to_input_idx_nine_by_nine(&coord);

        assert_eq!(40, idx);
    }

    #[test]
    fn test_map_coord_to_input_idx_eight_by_eight_a1() {
        let coord = "a1".parse::<Coordinate>().unwrap();
        let idx = map_coord_to_input_idx_eight_by_eight(&coord);

        assert_eq!(56, idx);
    }

    #[test]
    fn test_map_coord_to_input_idx_eight_by_eight_a8() {
        let coord = "a8".parse::<Coordinate>().unwrap();
        let idx = map_coord_to_input_idx_eight_by_eight(&coord);

        assert_eq!(0, idx);
    }

    #[test]
    fn test_map_coord_to_input_idx_eight_by_eight_h1() {
        let coord = "h1".parse::<Coordinate>().unwrap();
        let idx = map_coord_to_input_idx_eight_by_eight(&coord);

        assert_eq!(63, idx);
    }

    #[test]
    fn test_map_coord_to_input_idx_eight_by_eight_h8() {
        let coord = "h8".parse::<Coordinate>().unwrap();
        let idx = map_coord_to_input_idx_eight_by_eight(&coord);

        assert_eq!(7, idx);
    }

    #[test]
    fn test_map_coord_to_input_idx_eight_by_eight_e5() {
        let coord = "e5".parse::<Coordinate>().unwrap();
        let idx = map_coord_to_input_idx_eight_by_eight(&coord);

        assert_eq!(28, idx);
    }

    #[test]
    fn test_map_action_to_output_idx_pawn_a9() {
        let coord = "a9".parse::<Coordinate>().unwrap();
        let action = ActionExpanded::MovePawn(coord).into();
        let idx = map_action_to_output_idx(&action);

        assert_eq!(0, idx);
    }

    #[test]
    fn test_map_action_to_output_idx_pawn_i1() {
        let coord = "i1".parse::<Coordinate>().unwrap();
        let action = ActionExpanded::MovePawn(coord).into();
        let idx = map_action_to_output_idx(&action);

        assert_eq!(80, idx);
    }

    #[test]
    fn test_map_action_to_output_idx_vertical_wall_a8() {
        let coord = "a8".parse::<Coordinate>().unwrap();
        let action = ActionExpanded::PlaceVerticalWall(coord).into();
        let idx = map_action_to_output_idx(&action);

        assert_eq!(81, idx);
    }

    #[test]
    fn test_map_action_to_output_idx_vertical_wall_h1() {
        let coord = "h1".parse::<Coordinate>().unwrap();
        let action = ActionExpanded::PlaceVerticalWall(coord).into();
        let idx = map_action_to_output_idx(&action);

        assert_eq!(144, idx);
    }

    #[test]
    fn test_map_action_to_output_idx_horizontal_wall_a8() {
        let coord = "a8".parse::<Coordinate>().unwrap();
        let action = ActionExpanded::PlaceHorizontalWall(coord).into();
        let idx = map_action_to_output_idx(&action);

        assert_eq!(145, idx);
    }

    #[test]
    fn test_map_action_to_output_idx_horizontal_wall_h1() {
        let coord = "h1".parse::<Coordinate>().unwrap();
        let action = ActionExpanded::PlaceHorizontalWall(coord).into();
        let idx = map_action_to_output_idx(&action);

        assert_eq!(208, idx);
    }

    #[test]
    fn test_game_state_to_input_initial_p1() {
        let game_state = GameState::initial();

        let input = game_state_to_input(&game_state);

        assert_approximate_eq_slice(&map_to_input_vec(76, 4, &[], &[], 0, 0), &input)
    }

    #[test]
    fn test_game_state_to_input_initial_p2() {
        let mut game_state = GameState::initial();
        game_state.take_action(&"e2".parse::<Action>().unwrap());

        let input = game_state_to_input(&game_state);

        assert_approximate_eq_slice(&map_to_input_vec(76, 13, &[], &[], 0, 0), &input)
    }

    #[test]
    fn test_game_state_to_input_walls_remaining() {
        let mut game_state = GameState::initial();
        game_state.take_action(&"h1h".parse::<Action>().unwrap());

        let input = game_state_to_input(&game_state);

        assert_approximate_eq_slice(&map_to_input_vec(76, 4, &[], &[9, 10], 0, 1), &input)
    }

    #[test]
    fn test_game_state_to_input_walls_remaining_p2() {
        let mut game_state = GameState::initial();
        game_state.take_action(&"h1h".parse::<Action>().unwrap());
        game_state.take_action(&"d9".parse::<Action>().unwrap());

        let input = game_state_to_input(&game_state);

        assert_approximate_eq_slice(&map_to_input_vec(76, 3, &[], &[79, 80], 1, 0), &input)
    }

    #[test]
    fn test_game_state_to_input_vertical_walls() {
        let mut game_state = GameState::initial();
        game_state.take_action(&"e2".parse::<Action>().unwrap());
        game_state.take_action(&"c4v".parse::<Action>().unwrap());

        let input = game_state_to_input(&game_state);

        assert_approximate_eq_slice(&map_to_input_vec(67, 4, &[38, 47], &[], 0, 1), &input)
    }

    #[test]
    fn test_game_state_to_input_vertical_walls_p2() {
        let mut game_state = GameState::initial();
        game_state.take_action(&"e2".parse::<Action>().unwrap());
        game_state.take_action(&"c4v".parse::<Action>().unwrap());
        game_state.take_action(&"e3".parse::<Action>().unwrap());

        let input = game_state_to_input(&game_state);

        assert_approximate_eq_slice(&map_to_input_vec(76, 22, &[32, 41], &[], 1, 0), &input)
    }
}
