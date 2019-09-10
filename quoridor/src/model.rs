use model::analytics::ActionWithPolicy;
use model::node_metrics::NodeMetrics;
use model::model_info::ModelInfo;
use model::tensorflow::model::TensorflowModel;
use model::tensorflow::get_latest_model_info::get_latest_model_info;
use super::action::{Action,Coordinate};
use super::constants::{INPUT_H,INPUT_W,INPUT_C,OUTPUT_SIZE};
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

impl model::tensorflow::model::Mapper<GameState,Action> for Mapper {
    fn game_state_to_input(&self, game_state: &GameState) -> Vec<Vec<Vec<f64>>> {
        let result: Vec<Vec<Vec<f64>>> = Vec::with_capacity(6);

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
        let horizontal_wall_vec = map_board_to_arr_invertable(*horizontal_wall_placement_board, BoardType::VerticalWall, invert);

        let curr_num_walls_placed = if *p1_turn_to_move { p1_num_walls_placed } else { p2_num_walls_placed };
        let oppo_num_walls_placed = if *p1_turn_to_move { p2_num_walls_placed } else { p1_num_walls_placed };
        let curr_num_walls_placed_norm = (*curr_num_walls_placed as f64) / 10.0;
        let oppo_num_walls_placed_norm = (*oppo_num_walls_placed as f64) / 10.0;

        izip!(
            curr_pawn_board_vec.iter(),
            oppo_pawn_board_vec.iter(),
            vertical_wall_vec.iter(),
            horizontal_wall_vec.iter()
        )
        .enumerate()
        .fold(result, |mut r, (i, (curr_pawn, oppo_pawn, vw, hw))| {
            let column_idx = i % 9;

            if column_idx == 0 {
                r.push(Vec::with_capacity(9))
            }

            let column_vec = r.last_mut().unwrap();

            column_vec.push(vec!(*curr_pawn, *oppo_pawn, *vw, *hw, curr_num_walls_placed_norm, oppo_num_walls_placed_norm));

            r
        })
    }

    fn policy_metrics_to_expected_input(&self, policy_metrics: &NodeMetrics<Action>) -> Vec<f64> {
        let total_visits = policy_metrics.visits as f64 - 1.0;
        let result:[f64; 209] = policy_metrics.children_visits.iter().fold([0.0; 209], |mut r, p| {
            let input_idx = map_action_to_input_idx(&p.0);

            r[input_idx] = p.1 as f64 / total_visits;
            r
        });

        result.to_vec()
    }

    fn policy_to_valid_actions(&self, game_state: &GameState, policy_scores: &Vec<f64>) -> Vec<ActionWithPolicy<Action>> {
        let valid_pawn_moves = game_state.get_valid_pawn_move_actions().into_iter();
        let valid_vert_walls = game_state.get_valid_vertical_wall_actions().into_iter();
        let valid_horiz_walls = game_state.get_valid_horizontal_wall_actions().into_iter();
        let actions = valid_pawn_moves.chain(valid_vert_walls).chain(valid_horiz_walls);

        let valid_actions_with_policies: Vec<ActionWithPolicy<Action>> = actions
            .map(|a|
            {
                let p_idx = map_action_to_input_idx(&a);
                let p = policy_scores[p_idx];

                ActionWithPolicy::new(
                    a,
                    p
                )
            }).collect();

        valid_actions_with_policies
    }
}

fn map_action_to_input_idx(action: &Action) -> usize {
    let len_moves_inputs = 81;
    let len_wall_inputs = 64;

    match action {
        Action::MovePawn(coord) => map_coord_to_input_idx_nine_by_nine(coord),
        Action::PlaceVerticalWall(coord) => map_coord_to_input_idx_eight_by_eight(coord) + len_moves_inputs,
        Action::PlaceHorizontalWall(coord) => map_coord_to_input_idx_eight_by_eight(coord) + len_moves_inputs + len_wall_inputs
    }
}

fn map_coord_to_input_idx_nine_by_nine(coord: &Coordinate) -> usize {
    let col = match coord.column {
        'a' => 0,
        'b' => 1,
        'c' => 2,
        'd' => 3,
        'e' => 4,
        'f' => 5,
        'g' => 6,
        'h' => 7,
         _  => 8
    };

    col + ((9 - coord.row) * 9)
}

fn map_coord_to_input_idx_eight_by_eight(coord: &Coordinate) -> usize {
    let col = match coord.column {
        'a' => 0,
        'b' => 1,
        'c' => 2,
        'd' => 3,
        'e' => 4,
        'f' => 5,
        'g' => 6,
         _  => 7
    };

    col + ((8 - coord.row) * 8)
}

impl model::model::ModelFactory for ModelFactory {
    type M = TensorflowModel<GameState,Action,Engine,Mapper>;

    fn create(&self, model_info: &ModelInfo, num_filters: usize, num_blocks: usize) -> Self::M {
        TensorflowModel::<GameState,Action,Engine,Mapper>::create(
            model_info,
            num_filters,
            num_blocks,
            (INPUT_H, INPUT_W, INPUT_C),
            OUTPUT_SIZE
        ).unwrap();

        self.get(model_info)
    }

    fn get(&self, model_info: &ModelInfo) -> Self::M {
        let mapper = Mapper::new();

        TensorflowModel::new(
            model_info.clone(),
            Engine::new(),
            mapper
        )
    }

    fn get_latest(&self, model_info: &ModelInfo) -> Result<ModelInfo, Error> {
        Ok(get_latest_model_info(model_info)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
