// use model::analytics::ActionWithPolicy;
// use model::node_metrics::NodeMetrics;
// use model::model_info::ModelInfo;
// use model::tensorflow_serving::model::TensorflowServingModel;
// use model::tensorflow_serving::get_latest_model_info::get_latest_model_info;
// use super::action::Action;
// use super::engine::Engine;
// use super::engine::GameState;
// use super::board::{map_pawn_board_to_arr,map_pawn_board_to_arr_invert,map_wall_board_to_arr,map_wall_board_to_arr_invert};

// use itertools::izip;

// pub struct ModelFactory {}

// impl ModelFactory {
//     pub fn new() -> Self {
//         Self {}
//     }
// }

// pub struct Mapper {}

// impl Mapper {
//     fn new() -> Self {
//         Self {}
//     }
// }

// impl model::tensorflow_serving::model::Mapper<GameState,Action> for Mapper {
//     fn game_state_to_input(&self, game_state: &GameState) -> Vec<Vec<Vec<f64>>> {
//         let result: Vec<Vec<Vec<f64>>> = Vec::with_capacity(6);

//         let GameState {
//             p1_turn_to_move,
//             p1_pawn_board,
//             p2_pawn_board,
//             vertical_wall_placement_board,
//             horizontal_wall_placement_board,
//             p1_num_walls_placed,
//             p2_num_walls_placed
//          } = game_state;

//         let p1_pawn_board_vec = if *p1_turn_to_move { map_pawn_board_to_arr(*p1_pawn_board) } else { map_pawn_board_to_arr_invert(*p2_pawn_board) };
//         let p2_pawn_board_vec = if *p1_turn_to_move { map_pawn_board_to_arr(*p2_pawn_board) } else { map_pawn_board_to_arr_invert(*p1_pawn_board) };
//         let vertical_wall_placement_board = if *p1_turn_to_move { map_wall_board_to_arr(*vertical_wall_placement_board) } else { map_wall_board_to_arr_invert(*vertical_wall_placement_board) };
//         let horizontal_wall_placement_board = if *p1_turn_to_move { map_wall_board_to_arr(*horizontal_wall_placement_board) } else { map_wall_board_to_arr_invert(*horizontal_wall_placement_board) };
//         let p1_walls_placed = if *p1_turn_to_move { p1_num_walls_placed } else { p2_num_walls_placed };
//         let p2_walls_placed = if *p1_turn_to_move { p2_num_walls_placed } else { p1_num_walls_placed };

//         izip!(
//             p1_pawn_board_vec.iter(),
//             p2_pawn_board_vec.iter(),
//             vertical_wall_placement_board.iter(),
//             horizontal_wall_placement_board.iter()
//         )
//         .enumerate()
//         .fold(result, |mut r, (i, (p1, p2, vw, hw))| {
//             let column_idx = i % 9;

//             if column_idx == 0 {
//                 r.push(Vec::with_capacity(9))
//             }

//             let column_vec = r.last_mut().unwrap();

//             column_vec.push(vec!(*p1, *p2, *vw, *hw));

//             r
//         })
//     }

//     fn policy_metrics_to_expected_input(&self, policy_metrics: &NodeMetrics<Action>) -> Vec<f64> {
//         let total_visits = policy_metrics.visits as f64 - 1.0;
//         let result:[f64; 7] = policy_metrics.children_visits.iter().fold([0.0; 7], |mut r, p| {
//             match p.0 { Action::DropPiece(column) => r[column as usize - 1] = p.1 as f64 / total_visits };
//             r
//         });

//         result.to_vec()
//     }

//     fn policy_to_valid_actions(&self, game_state: &GameState, policy_scores: &Vec<f64>) -> Vec<ActionWithPolicy<Action>> {
//          let valid_actions_with_policies: Vec<ActionWithPolicy<Action>> = game_state.get_valid_actions().iter()
//             .zip(policy_scores).enumerate()
//             .filter_map(|(i, (v, p))|
//             {
//                 if *v {
//                     Some(ActionWithPolicy::new(
//                         Action::DropPiece((i + 1) as u64),
//                         *p
//                     ))
//                 } else {
//                     None
//                 }
//             }).collect();

//         valid_actions_with_policies
//     }
// }

// impl model::model::ModelFactory for ModelFactory {
//     type M = TensorflowServingModel<GameState,Action,Engine,Mapper>;

//     fn create(&self, model_info: &ModelInfo, num_filters: usize, num_blocks: usize) -> Self::M {
//         // @TODO: Replace with code to create the model.
//         let latest_model_info = get_latest_model_info(model_info).expect("Failed to get latest model");
//         let mapper = Mapper::new();

//         TensorflowServingModel::new(
//             latest_model_info,
//             Engine::new(),
//             mapper
//         )     
//     }

//     fn get(&self, model_info: &ModelInfo) -> Self::M {
//         let mapper = Mapper::new();

//         TensorflowServingModel::new(
//             model_info.clone(),
//             Engine::new(),
//             mapper
//         )
//     }

//     fn get_latest(&self, model_info: &ModelInfo) -> Self::M {
//         let latest_model_info = get_latest_model_info(model_info).expect("Failed to get latest model");
//         let mapper = Mapper::new();

//         TensorflowServingModel::new(
//             latest_model_info,
//             Engine::new(),
//             mapper
//         )
//     }
// }
