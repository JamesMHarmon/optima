use crate::BOARD_SIZE;

use super::transposition_entry::TranspositionEntry;
use half::f16;
use std::convert::TryInto;
use tensorflow_model::{InputMap, PolicyMap, TranspositionMap, ValueMap};

use super::constants::{
    ASCII_LETTER_A, BOARD_HEIGHT, BOARD_WIDTH, INPUT_C, INPUT_H, INPUT_W, NUM_WALLS_PER_PLAYER,
    OUTPUT_SIZE, PAWN_BOARD_SIZE, WALL_BOARD_SIZE,
};
use super::{Action, ActionType, Coordinate, GameState, Value};
use engine::Value as ValueTrait;
use model::logits::update_logit_policies_to_softmax;
use model::{ActionWithPolicy, BasicGameStateAnalysis, ConvInputBuilder, NodeMetrics};
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
        let valid_pawn_moves = game_state.valid_pawn_move_actions();
        let valid_vert_walls = game_state.valid_vertical_wall_actions();
        let valid_horiz_walls = game_state.valid_horizontal_wall_actions();
        let actions = valid_pawn_moves
            .chain(valid_vert_walls)
            .chain(valid_horiz_walls);

        let rotate = !game_state.p1_turn_to_move();
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
        let player_to_move = game_state.player_to_move();
        let val = value.get_value_for_player(player_to_move);
        (val * 2.0) - 1.0
    }

    fn map_value_output_to_value(&self, game_state: &GameState, value_output: f32) -> Value {
        let curr_val = (value_output + 1.0) / 2.0;
        let opp_val = 1.0 - curr_val;
        if game_state.p1_turn_to_move() {
            Value([curr_val, opp_val])
        } else {
            Value([opp_val, curr_val])
        }
    }
}

impl InputMap<GameState> for Mapper {
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
    let coord = action.coord();

    match action.action_type() {
        ActionType::PawnMove => map_coord_to_output_idx_nine_by_nine(&coord),
        ActionType::VerticalWall => {
            map_coord_to_output_idx_eight_by_eight(&coord) + len_moves_inputs
        }
        ActionType::HorizontalWall => {
            map_coord_to_output_idx_eight_by_eight(&coord) + len_moves_inputs + len_wall_inputs
        }
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
