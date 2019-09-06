use model::analytics::ActionWithPolicy;
use model::node_metrics::NodeMetrics;
use model::model_info::ModelInfo;
use model::tensorflow::model::TensorflowModel;
use model::tensorflow::get_latest_model_info::get_latest_model_info;
use super::constants::{INPUT_H,INPUT_W,INPUT_C,OUTPUT_SIZE};
use super::action::Action;
use super::engine::Engine;
use super::engine::GameState;
use super::board::map_board_to_arr;

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

        map_board_to_arr(game_state.p1_piece_board).iter()
            .zip(map_board_to_arr(game_state.p2_piece_board).iter())
            .enumerate()
            .fold(result, |mut r, (i, (p1, p2))| {
                let column_idx = i % 7;
                
                if column_idx == 0 {
                    r.push(Vec::with_capacity(7))
                }

                let column_vec = r.last_mut().unwrap();

                // The input is normalized by listing the player to move first. This is different than having
                // black first and then red. So on red's turn, red will be listed first, then black.
                let (c1, c2) = if game_state.p1_turn_to_move {
                    (*p1, *p2)
                } else {
                    (*p2, *p1)
                };

                column_vec.push(vec!(c1, c2));

                r
            })
    }

    fn policy_metrics_to_expected_input(&self, policy_metrics: &NodeMetrics<Action>) -> Vec<f64> {
        let total_visits = policy_metrics.visits as f64 - 1.0;
        let result:[f64; 7] = policy_metrics.children_visits.iter().fold([0.0; 7], |mut r, p| {
            match p.0 { Action::DropPiece(column) => r[column as usize - 1] = p.1 as f64 / total_visits };
            r
        });

        result.to_vec()
    }

    fn policy_to_valid_actions(&self, game_state: &GameState, policy_scores: &Vec<f64>) -> Vec<ActionWithPolicy<Action>> {
         let valid_actions_with_policies: Vec<ActionWithPolicy<Action>> = game_state.get_valid_actions().iter()
            .zip(policy_scores).enumerate()
            .filter_map(|(i, (v, p))|
            {
                if *v {
                    Some(ActionWithPolicy::new(
                        Action::DropPiece((i + 1) as u64),
                        *p
                    ))
                } else {
                    None
                }
            }).collect();

        valid_actions_with_policies
    }
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

    fn get_latest(&self, model_info: &ModelInfo) -> Self::M {
        let latest_model_info = get_latest_model_info(model_info).expect("Failed to get latest model");
        let mapper = Mapper::new();

        TensorflowModel::new(
            latest_model_info,
            Engine::new(),
            mapper
        )
    }
}
