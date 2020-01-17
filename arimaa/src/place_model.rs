use model::position_metrics::PositionMetrics;
use model::model::ModelOptions;
use model::analytics::ActionWithPolicy;
use model::node_metrics::NodeMetrics;
use model::model_info::ModelInfo;
use model::tensorflow::model::{TensorflowModel,TensorflowModelOptions};
use model::tensorflow::get_latest_model_info::get_latest_model_info;
use engine::value::{Value as ValueTrait};
use super::board::set_placement_board_bits;
use super::value::Value;
use super::action::{Action,Piece};
use super::constants::{PLACE_INPUT_H as INPUT_H,PLACE_INPUT_W as INPUT_W,PLACE_INPUT_C as INPUT_C,PLACE_OUTPUT_SIZE as OUTPUT_SIZE,PLACE_MOVES_LEFT_SIZE as MOVES_LEFT_SIZE,PLACE_INPUT_SIZE as INPUT_SIZE,* };
use super::engine::Engine;
use super::engine::GameState;

use failure::Error;

/*
    Layers:
    In:
    6 piece boards
    6 curr pieces remaining temp board
    1 player
    1 piece placement bit
    Out:
    6 pieces
*/
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
        let mut input: Vec<f32> = Vec::with_capacity(INPUT_SIZE);
        input.extend(std::iter::repeat(0.0).take(INPUT_SIZE));

        let is_p1_turn_to_move = game_state.is_p1_turn_to_move();
        let piece_board = game_state.get_piece_board();
        let player_piece_board = piece_board.get_player_piece_mask(is_p1_turn_to_move);

        for (i, piece) in [Piece::Elephant, Piece::Camel, Piece::Horse, Piece::Dog, Piece::Cat, Piece::Rabbit].iter().enumerate() {
            let piece_bits = piece_board.get_bits_by_piece_type(*piece);
            let offset = i;

            insert_input_channel_bits(&mut input, offset, piece_bits);

            let num_pieces_placed = (piece_bits & player_piece_board).count_ones();
            let num_pieces_to_place = match piece {
                Piece::Elephant | Piece::Camel => 1,
                Piece::Horse | Piece::Dog | Piece::Cat => 2,
                Piece::Rabbit => 8
            };

            let offset = NUM_PIECE_TYPES + i;
            let num_piece_remaining = num_pieces_placed as f32 / num_pieces_to_place as f32;
            insert_input_channel_bit(&mut input, offset, num_piece_remaining);
        }

        let offset = NUM_PIECE_TYPES + NUM_PIECE_TYPES;
        let placement_bit = piece_board.get_placement_bit();
        insert_input_channel_bits(&mut input, offset, placement_bit);

        let offset = NUM_PIECE_TYPES + NUM_PIECE_TYPES + PLACEMENT_BIT_CHANNEL;
        let is_p1_turn_value = if is_p1_turn_to_move { 0.0 } else { 1.0 };
        insert_input_channel_bit(&mut input, offset, is_p1_turn_value);

        input
    }

    fn get_symmetries(&self, metrics: PositionMetrics<GameState,Action,Value>) -> Vec<PositionMetrics<GameState,Action,Value>> {
        vec!(metrics)
    }

    fn get_input_dimensions(&self) -> [u64; 3] {
        [INPUT_H as u64, INPUT_W as u64, INPUT_C as u64]
    }

    fn policy_metrics_to_expected_output(&self, _game_state: &GameState, policy_metrics: &NodeMetrics<Action>) -> Vec<f32> {
        let total_visits = policy_metrics.visits as f32 - 1.0;
        let mut inputs = Vec::with_capacity(OUTPUT_SIZE);
        inputs.extend(std::iter::repeat(0.0).take(OUTPUT_SIZE));

        policy_metrics.children.iter().fold(inputs, |mut r, (action, _w, visits)| {
            let policy_index = map_action_to_policy_output_idx(action);

            r[policy_index] = *visits as f32 / total_visits;
            r
        })
    }

    fn policy_to_valid_actions(&self, game_state: &GameState, policy_scores: &[f32]) -> Vec<ActionWithPolicy<Action>> {
        let mut valid_actions_with_policies: Vec<_> = game_state.valid_actions().into_iter()
            .map(|action| {
                let policy_index = map_action_to_policy_output_idx(&action);
                let policy_score = policy_scores[policy_index];

                ActionWithPolicy::new(
                    action,
                    policy_score
                )
            }).collect();

        let policy_sum: f32 = valid_actions_with_policies.iter().map(|v| v.policy_score).sum();

        let policy_scale = 1.0 / policy_sum;

        for awp in valid_actions_with_policies.iter_mut() {
            awp.policy_score = awp.policy_score * policy_scale;
        }

        valid_actions_with_policies
    }

    fn map_value_output_to_value(&self, game_state: &GameState, value_output: f32) -> Value {
        let curr_val = (value_output + 1.0) / 2.0;
        let opp_val = 1.0 - curr_val;
        if game_state.is_p1_turn_to_move() { Value([curr_val, opp_val]) } else { Value([opp_val, curr_val]) }
    }

    fn map_value_to_value_output(&self, game_state: &GameState, value: &Value) -> f32 {
        let player_to_move = if game_state.is_p1_turn_to_move() { 1 } else { 2 };
        let val = value.get_value_for_player(player_to_move);
        (val * 2.0) - 1.0
    }
}

fn map_action_to_policy_output_idx(action: &Action) -> usize {
    match action {
        Action::Place(piece) => match piece {
            Piece::Elephant => 0,
            Piece::Camel => 1,
            Piece::Horse => 2,
            Piece::Dog => 3,
            Piece::Cat => 4,
            Piece::Rabbit => 5
        },
        _ => panic!("Action not expected")
    }
}

fn insert_input_channel_bit(input: &mut [f32], offset: usize, value: f32) {
    for board_idx in 0..PLACE_BOARD_SIZE {
        let cell_idx = board_idx * INPUT_C + offset;
        input[cell_idx] = value;
    }
}

fn insert_input_channel_bits(input: &mut [f32], offset: usize, bits: u64) {
    set_placement_board_bits(input, offset, bits);
}

impl model::model::ModelFactory for ModelFactory {
    type M = TensorflowModel<Engine,Mapper>;
    type O = ModelOptions;

    fn create(&self, model_info: &ModelInfo, options: &Self::O) -> Self::M
    {
        TensorflowModel::<Engine,Mapper>::create(
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
