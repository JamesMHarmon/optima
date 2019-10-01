use model::analysis_cache::cache;
use model::analytics::ActionWithPolicy;
use model::node_metrics::NodeMetrics;
use model::model_info::ModelInfo;
use model::analysis_cache::AnalysisCacheModel;
use model::tensorflow::model::TensorflowModel;
use model::tensorflow::get_latest_model_info::get_latest_model_info;
use super::constants::{ACTIONS_TO_CACHE,INPUT_H,INPUT_W,INPUT_C,OUTPUT_SIZE};
use super::action::Action;
use super::engine::Engine;
use super::engine::GameState;
use super::board::map_board_to_arr;

use failure::Error;

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
    fn game_state_to_input(&self, game_state: &GameState) -> Vec<f32> {
        let mut result: Vec<f32> = Vec::with_capacity(INPUT_H * INPUT_W * INPUT_C);
        let (curr_piece_board, opp_piece_board) = if game_state.p1_turn_to_move {
            (game_state.p1_piece_board, game_state.p2_piece_board)
        } else {
            (game_state.p2_piece_board, game_state.p1_piece_board)
        };

        for (curr, opp) in map_board_to_arr(curr_piece_board).iter()
            .zip(map_board_to_arr(opp_piece_board).iter()) {
                result.push(*curr);
                result.push(*opp);
            }

        result
    }

    fn get_input_dimensions(&self) -> [u64; 3] {
        [INPUT_H as u64, INPUT_W as u64, INPUT_C as u64]
    }

    fn policy_metrics_to_expected_output(&self, _game_state: &GameState, policy_metrics: &NodeMetrics<Action>) -> Vec<f32> {
        let total_visits = policy_metrics.visits as f32 - 1.0;
        let result:[f32; 7] = policy_metrics.children_visits.iter().fold([0.0; 7], |mut r, p| {
            match p.0 { Action::DropPiece(column) => r[column as usize - 1] = p.1 as f32 / total_visits };
            r
        });

        result.to_vec()
    }

    fn policy_to_valid_actions(&self, game_state: &GameState, policy_scores: &[f32]) -> Vec<ActionWithPolicy<Action>> {
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
    type M = AnalysisCacheModel<ShouldCache,TensorflowModel<Engine,Mapper>>;

    fn create(&self, model_info: &ModelInfo, num_filters: usize, num_blocks: usize) -> Self::M {
        TensorflowModel::<Engine,Mapper>::create(
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

        cache(
            TensorflowModel::new(
                model_info.clone(),
                Engine::new(),
                mapper
            )
        )
    }

    fn get_latest(&self, model_info: &ModelInfo) -> Result<ModelInfo,Error> {
        Ok(get_latest_model_info(model_info)?)
    }
}

pub struct ShouldCache {}

impl model::analysis_cache::ShouldCache for ShouldCache {
    type State = GameState;

    fn should_cache(game_state: &GameState) -> bool {
        game_state.number_of_actions() <= ACTIONS_TO_CACHE
    }
}
