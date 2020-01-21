use model::logits::update_logit_policies_to_softmax;
use model::position_metrics::PositionMetrics;
use model::model::ModelOptions;
use model::analysis_cache::{cache,AnalysisCacheModel};
use model::analytics::ActionWithPolicy;
use model::node_metrics::NodeMetrics;
use model::model_info::ModelInfo;
use model::tensorflow::model::{TensorflowModel,TensorflowModelOptions};
use model::tensorflow::get_latest_model_info::get_latest_model_info;
use engine::value::{Value as ValueTrait};
use super::value::Value;
use super::action::{Action,Coordinate};
use super::constants::{ASCII_LETTER_A,INPUT_H,INPUT_W,INPUT_C,OUTPUT_SIZE,MOVES_LEFT_SIZE,MAX_WALLS_PLACED_TO_CACHE,NUM_WALLS_PER_PLAYER,WALL_BOARD_SIZE,PAWN_BOARD_SIZE,BOARD_WIDTH,BOARD_HEIGHT};
use super::engine::Engine;
use super::engine::GameState;
use super::board::{map_board_to_arr_invertable,BoardType};

use failure::Error;
use itertools::izip;

pub struct ModelFactory {}

impl ModelFactory {
    pub fn new() -> Self {
        Self {}
    }
}

pub struct Mapper {}

impl Mapper {
    fn new() -> Self {
        Self {}
    }
}

impl model::tensorflow::model::Mapper<GameState,Action,Value> for Mapper {
    fn game_state_to_input(&self, game_state: &GameState) -> Vec<f32> {
        let mut result: Vec<f32> = Vec::with_capacity(INPUT_H * INPUT_W * INPUT_C);

        let GameState {
            p1_turn_to_move,
            p1_pawn_board,
            p2_pawn_board,
            vertical_wall_placement_board,
            horizontal_wall_placement_board,
            p1_num_walls_placed,
            p2_num_walls_placed,
            ..
        } = game_state;

        let curr_player_pawn_board = if *p1_turn_to_move { *p1_pawn_board } else { *p2_pawn_board };
        let oppo_player_pawn_board = if *p1_turn_to_move { *p2_pawn_board } else { *p1_pawn_board };

        let invert = !*p1_turn_to_move;

        let curr_pawn_board_vec = map_board_to_arr_invertable(curr_player_pawn_board, BoardType::Pawn, invert);
        let oppo_pawn_board_vec = map_board_to_arr_invertable(oppo_player_pawn_board, BoardType::Pawn, invert);
        let vertical_wall_vec = map_board_to_arr_invertable(*vertical_wall_placement_board, BoardType::VerticalWall, invert);
        let horizontal_wall_vec = map_board_to_arr_invertable(*horizontal_wall_placement_board, BoardType::HorizontalWall, invert);

        let curr_num_walls_placed = if *p1_turn_to_move { p1_num_walls_placed } else { p2_num_walls_placed };
        let oppo_num_walls_placed = if *p1_turn_to_move { p2_num_walls_placed } else { p1_num_walls_placed };
        let curr_num_walls_placed_norm = (*curr_num_walls_placed as f32) / NUM_WALLS_PER_PLAYER as f32;
        let oppo_num_walls_placed_norm = (*oppo_num_walls_placed as f32) / NUM_WALLS_PER_PLAYER as f32;

        for (curr_pawn, oppo_pawn, vw, hw) in izip!(
            curr_pawn_board_vec.iter(),
            oppo_pawn_board_vec.iter(),
            vertical_wall_vec.iter(),
            horizontal_wall_vec.iter()
        ) {
            result.push(*curr_pawn);
            result.push(*oppo_pawn);
            result.push(*vw);
            result.push(*hw);
            result.push(curr_num_walls_placed_norm);
            result.push(oppo_num_walls_placed_norm);
        }

        result
    }

    fn get_input_dimensions(&self) -> [u64; 3] {
        [INPUT_H as u64, INPUT_W as u64, INPUT_C as u64]
    }

    fn get_symmetries(&self, metrics: PositionMetrics<GameState,Action,Value>) -> Vec<PositionMetrics<GameState,Action,Value>> {
        //@TODO: Add symmetries for Quoridor
        vec!(metrics)
    }

    fn policy_metrics_to_expected_output(&self, game_state: &GameState, policy_metrics: &NodeMetrics<Action>) -> Vec<f32> {
        let total_visits = policy_metrics.visits as f32 - 1.0;
        let invert = !game_state.p1_turn_to_move;
        let mut inputs = Vec::with_capacity(OUTPUT_SIZE);
        inputs.extend(std::iter::repeat(0.0).take(OUTPUT_SIZE));

        policy_metrics.children.iter().fold(inputs, |mut r, (action, _w, visits)| {
            // Policy scores for quoridor should be in the perspective of player 1. That means that if we are p2, we need to flip the actions as if we were looking
            // at the board from the perspective of player 1, but with the pieces inverted.
            let input_idx = if invert {
                map_action_to_input_idx(&action.invert())
            } else {
                map_action_to_input_idx(&action)
            };

            r[input_idx] = *visits as f32 / total_visits;
            r
        })
    }

    fn policy_to_valid_actions(&self, game_state: &GameState, policy_scores: &[f32]) -> Vec<ActionWithPolicy<Action>> {
        let valid_pawn_moves = game_state.get_valid_pawn_move_actions().into_iter();
        let valid_vert_walls = game_state.get_valid_vertical_wall_actions().into_iter();
        let valid_horiz_walls = game_state.get_valid_horizontal_wall_actions().into_iter();
        let actions = valid_pawn_moves.chain(valid_vert_walls).chain(valid_horiz_walls);
        let invert = !game_state.p1_turn_to_move;

        let mut valid_actions_with_policies: Vec<ActionWithPolicy<Action>> = actions
            .map(|a|
            {
                // Policy scores coming from the quoridor model are always from the perspective of player 1.
                // This means that if we are p2, we need to flip the actions coming back and translate them
                // to be actions in the p2 perspective.
                let p_idx = if invert {
                    map_action_to_input_idx(&a.invert())
                } else {
                    map_action_to_input_idx(&a)
                };

                let p = policy_scores[p_idx];

                ActionWithPolicy::new(
                    a,
                    p
                )
            }).collect();

        update_logit_policies_to_softmax(&mut valid_actions_with_policies);

        valid_actions_with_policies
    }

    fn map_value_output_to_value(&self, game_state: &GameState, value_output: f32) -> Value {
        let curr_val = (value_output + 1.0) / 2.0;
        let opp_val = 1.0 - curr_val;
        if game_state.p1_turn_to_move { Value([curr_val, opp_val]) } else { Value([opp_val, curr_val]) }
    }

    fn map_value_to_value_output(&self, game_state: &GameState, value: &Value) -> f32 {
        let player_to_move = if game_state.p1_turn_to_move { 1 } else { 2 };
        let val = value.get_value_for_player(player_to_move);
        (val * 2.0) - 1.0
    }
}

fn map_action_to_input_idx(action: &Action) -> usize {
    let len_moves_inputs = PAWN_BOARD_SIZE;
    let len_wall_inputs = WALL_BOARD_SIZE;

    match action {
        Action::MovePawn(coord) => map_coord_to_input_idx_nine_by_nine(coord),
        Action::PlaceVerticalWall(coord) => map_coord_to_input_idx_eight_by_eight(coord) + len_moves_inputs,
        Action::PlaceHorizontalWall(coord) => map_coord_to_input_idx_eight_by_eight(coord) + len_moves_inputs + len_wall_inputs
    }
}

fn map_coord_to_input_idx_nine_by_nine(coord: &Coordinate) -> usize {
    let col_idx = (coord.column as u8 - ASCII_LETTER_A) as usize;

    col_idx + ((BOARD_HEIGHT - coord.row) * BOARD_WIDTH)
}

fn map_coord_to_input_idx_eight_by_eight(coord: &Coordinate) -> usize {
    let col_idx = (coord.column as u8 - ASCII_LETTER_A) as usize;

    col_idx + ((BOARD_HEIGHT - coord.row - 1) * (BOARD_WIDTH - 1))
}

impl model::model::ModelFactory for ModelFactory {
    type M = AnalysisCacheModel<ShouldCache,TensorflowModel<GameState,Action,Value,Engine,Mapper>>;
    type O = ModelOptions;

    fn create(&self, model_info: &ModelInfo, options: &Self::O) -> Self::M
    {
        TensorflowModel::<GameState,Action,Value,Engine,Mapper>::create(
            model_info,
            &TensorflowModelOptions {
                num_filters: options.number_of_filters,
                num_blocks: options.number_of_residual_blocks,
                channel_height: INPUT_H,
                channel_width: INPUT_W,
                channels: INPUT_C,
                output_size: OUTPUT_SIZE,
                moves_left_size: MOVES_LEFT_SIZE
            }
        ).unwrap();

        self.get(model_info)
    }

    fn get(&self, model_info: &ModelInfo) -> Self::M {
        let mapper = Mapper::new();

        cache(
            TensorflowModel::new(
                model_info.clone(),
                Engine::new(),
                mapper
            )
        )
    }

    fn get_latest(&self, model_info: &ModelInfo) -> Result<ModelInfo, Error> {
        Ok(get_latest_model_info(model_info)?)
    }
}

pub struct ShouldCache {}

impl model::analysis_cache::ShouldCache for ShouldCache {
    type State = GameState;

    fn should_cache(game_state: &GameState) -> bool {
        game_state.p1_num_walls_placed + game_state.p2_num_walls_placed <= MAX_WALLS_PLACED_TO_CACHE
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::game_state::{GameState as GameStateTrait};
    use model::tensorflow::model::{Mapper as MapperTrait};

    fn map_to_input_vec(curr_pawn_idx: usize, opp_pawn_idx: usize, vertical_wall_idxs: &[usize], horizontal_wall_idxs: &[usize], curr_num_walls_placed: usize, opp_num_walls_placed: usize) -> Vec<f32> {
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

    #[test]
    fn test_map_coord_to_input_idx_nine_by_nine_a1() {
        let coord = Coordinate::new('a', 1);
        let idx = map_coord_to_input_idx_nine_by_nine(&coord);

        assert_eq!(72, idx);
    }

    #[test]
    fn test_map_coord_to_input_idx_nine_by_nine_a9() {
        let coord = Coordinate::new('a', 9);
        let idx = map_coord_to_input_idx_nine_by_nine(&coord);

        assert_eq!(0, idx);
    }

    #[test]
    fn test_map_coord_to_input_idx_nine_by_nine_i1() {
        let coord = Coordinate::new('i', 1);
        let idx = map_coord_to_input_idx_nine_by_nine(&coord);

        assert_eq!(80, idx);
    }

    #[test]
    fn test_map_coord_to_input_idx_nine_by_nine_i9() {
        let coord = Coordinate::new('i', 9);
        let idx = map_coord_to_input_idx_nine_by_nine(&coord);

        assert_eq!(8, idx);
    }

    #[test]
    fn test_map_coord_to_input_idx_nine_by_nine_e5() {
        let coord = Coordinate::new('e', 5);
        let idx = map_coord_to_input_idx_nine_by_nine(&coord);

        assert_eq!(40, idx);
    }

    #[test]
    fn test_map_coord_to_input_idx_eight_by_eight_a1() {
        let coord = Coordinate::new('a', 1);
        let idx = map_coord_to_input_idx_eight_by_eight(&coord);

        assert_eq!(56, idx);
    }

    #[test]
    fn test_map_coord_to_input_idx_eight_by_eight_a8() {
        let coord = Coordinate::new('a', 8);
        let idx = map_coord_to_input_idx_eight_by_eight(&coord);

        assert_eq!(0, idx);
    }

    #[test]
    fn test_map_coord_to_input_idx_eight_by_eight_h1() {
        let coord = Coordinate::new('h', 1);
        let idx = map_coord_to_input_idx_eight_by_eight(&coord);

        assert_eq!(63, idx);
    }

    #[test]
    fn test_map_coord_to_input_idx_eight_by_eight_h8() {
        let coord = Coordinate::new('h', 8);
        let idx = map_coord_to_input_idx_eight_by_eight(&coord);

        assert_eq!(7, idx);
    }

    #[test]
    fn test_map_coord_to_input_idx_eight_by_eight_e5() {
        let coord = Coordinate::new('e', 5);
        let idx = map_coord_to_input_idx_eight_by_eight(&coord);

        assert_eq!(28, idx);
    }

    #[test]
    fn test_map_action_to_input_idx_pawn_a9() {
        let coord = Coordinate::new('a', 9);
        let action = Action::MovePawn(coord);
        let idx = map_action_to_input_idx(&action);

        assert_eq!(0, idx);
    }

    #[test]
    fn test_map_action_to_input_idx_pawn_i1() {
        let coord = Coordinate::new('i', 1);
        let action = Action::MovePawn(coord);
        let idx = map_action_to_input_idx(&action);

        assert_eq!(80, idx);
    }

    #[test]
    fn test_map_action_to_input_idx_vertical_wall_a8() {
        let coord = Coordinate::new('a', 8);
        let action = Action::PlaceVerticalWall(coord);
        let idx = map_action_to_input_idx(&action);

        assert_eq!(81, idx);
    }

    #[test]
    fn test_map_action_to_input_idx_vertical_wall_h1() {
        let coord = Coordinate::new('h', 1);
        let action = Action::PlaceVerticalWall(coord);
        let idx = map_action_to_input_idx(&action);

        assert_eq!(144, idx);
    }

    #[test]
    fn test_map_action_to_input_idx_horizontal_wall_a8() {
        let coord = Coordinate::new('a', 8);
        let action = Action::PlaceHorizontalWall(coord);
        let idx = map_action_to_input_idx(&action);

        assert_eq!(145, idx);
    }

    #[test]
    fn test_map_action_to_input_idx_horizontal_wall_h1() {
        let coord = Coordinate::new('h', 1);
        let action = Action::PlaceHorizontalWall(coord);
        let idx = map_action_to_input_idx(&action);

        assert_eq!(208, idx);
    }

    #[test]
    fn test_game_state_to_input_initial_p1() {
        let mapper = Mapper::new();

        let game_state = GameState::initial();

        let input = mapper.game_state_to_input(&game_state);

        assert_eq!(
            map_to_input_vec(76, 4, &vec!(), &vec!(), 0, 0),
            input
        )
    }

    #[test]
    fn test_game_state_to_input_initial_p2() {
        let mapper = Mapper::new();

        let game_state = GameState::initial();
        let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('e', 2)));

        let input = mapper.game_state_to_input(&game_state);

        assert_eq!(
            map_to_input_vec(76, 13, &vec!(), &vec!(), 0, 0),
            input
        )
    }

    #[test]
    fn test_game_state_to_input_walls_remaining() {
        let mapper = Mapper::new();

        let game_state = GameState::initial();
        let game_state = game_state.take_action(&Action::PlaceHorizontalWall(Coordinate::new('h', 1)));

        let input = mapper.game_state_to_input(&game_state);

        assert_eq!(
            map_to_input_vec(76, 4, &vec!(), &vec!(9,10), 0, 1),
            input
        )
    }

    #[test]
    fn test_game_state_to_input_walls_remaining_p2() {
        let mapper = Mapper::new();

        let game_state = GameState::initial();
        let game_state = game_state.take_action(&Action::PlaceHorizontalWall(Coordinate::new('h', 1)));
        let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('d', 9)));

        let input = mapper.game_state_to_input(&game_state);

        assert_eq!(
            map_to_input_vec(76, 3, &vec!(), &vec!(79,80), 1, 0),
            input
        )
    }

    #[test]
    fn test_game_state_to_input_vertical_walls() {
        let mapper = Mapper::new();

        let game_state = GameState::initial();
        let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('e', 2)));
        let game_state = game_state.take_action(&Action::PlaceVerticalWall(Coordinate::new('c', 4)));

        let input = mapper.game_state_to_input(&game_state);

        assert_eq!(
            map_to_input_vec(67, 4, &vec!(38,47), &vec!(), 0, 1),
            input
        )
    }

    #[test]
    fn test_game_state_to_input_vertical_walls_p2() {
        let mapper = Mapper::new();

        let game_state = GameState::initial();
        let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('e', 2)));
        let game_state = game_state.take_action(&Action::PlaceVerticalWall(Coordinate::new('c', 4)));
        let game_state = game_state.take_action(&Action::MovePawn(Coordinate::new('e', 3)));

        let input = mapper.game_state_to_input(&game_state);

        assert_eq!(
            map_to_input_vec(76, 22, &vec!(32,41), &vec!(), 1, 0),
            input
        )
    }
}
