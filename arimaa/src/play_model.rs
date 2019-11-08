use model::model::ModelOptions;
use model::analytics::ActionWithPolicy;
use model::node_metrics::NodeMetrics;
use model::model_info::ModelInfo;
use model::tensorflow::model::{TensorflowModel,TensorflowModelOptions};
use model::tensorflow::get_latest_model_info::get_latest_model_info;
use engine::value::{Value as ValueTrait};
use super::board::set_board_bits_invertable;
use super::value::Value;
use super::action::{Action,Square,Direction,Piece};
use super::constants::{PLAY_INPUT_H as INPUT_H,PLAY_INPUT_W as INPUT_W,PLAY_INPUT_C as INPUT_C,PLAY_OUTPUT_SIZE as OUTPUT_SIZE,PLAY_INPUT_SIZE as INPUT_SIZE,* };
use super::engine::Engine;
use super::engine::GameState;

use failure::Error;

/*
    Layers:
    In:
    6 curr piece boards
    6 opp piece boards
    x 4 (position at start of move, position after first action, second action, third action take)
    1 current step
    1 trap squares

    Out:
    4 directional boards (substract irrelevant squares)
    1 pass bit
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
        let mut result: Vec<f32> = Vec::with_capacity(INPUT_SIZE);
        result.extend(std::iter::repeat(0.0).take(INPUT_SIZE));

        set_board_state_squares(&mut result, game_state);

        set_step_num_squares(&mut result, game_state);

        set_trap_squares(&mut result);

        result
    }

    fn get_input_dimensions(&self) -> [u64; 3] {
        [INPUT_H as u64, INPUT_W as u64, INPUT_C as u64]
    }

    fn policy_metrics_to_expected_output(&self, game_state: &GameState, policy_metrics: &NodeMetrics<Action>) -> Vec<f32> {
        let total_visits = policy_metrics.visits as f32 - 1.0;
        let invert = !game_state.is_p1_turn_to_move();
        let mut inputs = Vec::with_capacity(OUTPUT_SIZE);
        inputs.extend(std::iter::repeat(0.0).take(OUTPUT_SIZE));

        policy_metrics.children_visits.iter().fold(inputs, |mut r, (action, visits)| {
            // Policy scores are in the perspective of player 1. That means that if we are p2, we need to flip the actions as if we were looking
            // at the board from the perspective of player 1, but with the pieces inverted.
            let policy_index = if invert {
                map_action_to_policy_output_idx(&action.invert())
            } else {
                map_action_to_policy_output_idx(action)
            };

            r[policy_index] = *visits as f32 / total_visits;
            r
        })
    }

    fn policy_to_valid_actions(&self, game_state: &GameState, policy_scores: &[f32]) -> Vec<ActionWithPolicy<Action>> {
        let invert = !game_state.is_p1_turn_to_move();

        let valid_actions_with_policies: Vec<_> = game_state.valid_actions().into_iter()
            .map(|action|
            {
                // Policy scores coming from the model are always from the perspective of player 1.
                // This means that if we are p2, we need to flip the actions coming back and translate them
                // to be actions in the p2 perspective.
                let policy_index = if invert {
                    map_action_to_policy_output_idx(&action.invert())
                } else {
                    map_action_to_policy_output_idx(&action)
                };

                let policy_score = policy_scores[policy_index];

                ActionWithPolicy::new(
                    action,
                    policy_score
                )
            }).collect();

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

fn set_board_state_squares(input: &mut [f32], game_state: &GameState) {
    let current_step_num = game_state.get_current_step();
    let is_p1_turn_to_move = game_state.is_p1_turn_to_move();
    let invert = !is_p1_turn_to_move;

    for (i, step) in (0..=current_step_num).rev().enumerate() {
        let piece_board = game_state.get_piece_board_for_step(step);
        let step_offset = i * BOARDS_PER_STATE;

        for (j, player) in [is_p1_turn_to_move, !is_p1_turn_to_move].into_iter().enumerate() {
            let player_offset = j * NUM_PIECE_TYPES;

            for (piece_offset, piece) in [Piece::Elephant, Piece::Camel, Piece::Horse, Piece::Dog, Piece::Cat, Piece::Rabbit].into_iter().enumerate() {
                let piece_bits = piece_board.get_bits_for_piece(piece, *player);

                let offset = step_offset + player_offset + piece_offset;
                set_board_bits_invertable(input, offset, piece_bits, invert);
            }
        }
    }
}

fn set_step_num_squares(input: &mut [f32], game_state: &GameState) {
    let current_step_num = game_state.get_current_step();
    let step_num_normalized = (current_step_num as f32) / (MAX_NUM_STEPS - 1) as f32;
    let step_num_offset = STEP_NUM_CHANNEL_IDX;
    for board_idx in 0..BOARD_SIZE {
        let cell_idx = board_idx * PLAY_INPUT_C + step_num_offset;
        input[cell_idx] = step_num_normalized;
    }
}

fn set_trap_squares(input: &mut [f32]) {
    input[INPUT_C * 18 + TRAP_CHANNEL_IDX] = 1.0;
    input[INPUT_C * 21 + TRAP_CHANNEL_IDX] = 1.0;
    input[INPUT_C * 42 + TRAP_CHANNEL_IDX] = 1.0;
    input[INPUT_C * 45 + TRAP_CHANNEL_IDX] = 1.0;
}


fn map_action_to_policy_output_idx(action: &Action) -> usize {
    match action {
        Action::Move(square, direction) => match direction {
            Direction::Up => map_coord_to_policy_output_idx_up(square),
            Direction::Right => map_coord_to_policy_output_idx_right(square) + NUM_UP_MOVES,
            Direction::Down => map_coord_to_policy_output_idx_down(square) + NUM_UP_MOVES + NUM_RIGHT_MOVES,
            Direction::Left => map_coord_to_policy_output_idx_left(square) + NUM_UP_MOVES + NUM_RIGHT_MOVES + NUM_DOWN_MOVES,
        },
        Action::Pass => NUM_UP_MOVES + NUM_RIGHT_MOVES + NUM_DOWN_MOVES + NUM_LEFT_MOVES,
        _ => panic!("Action not expected")
    }
}

fn map_coord_to_policy_output_idx_up(square: &Square) -> usize {
    square.get_index() as usize - BOARD_WIDTH
}

fn map_coord_to_policy_output_idx_right(square: &Square) -> usize {
    let square_index = square.get_index() as usize;
    let num_squares_to_skip = square_index / BOARD_WIDTH;
    square_index - num_squares_to_skip
}

fn map_coord_to_policy_output_idx_down(square: &Square) -> usize {
    square.get_index() as usize
}

fn map_coord_to_policy_output_idx_left(square: &Square) -> usize {
    let square_index = square.get_index() as usize;
    let num_squares_to_skip = square_index / BOARD_WIDTH;
    square_index - num_squares_to_skip - 1
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
                output_size: OUTPUT_SIZE
            }
        ).unwrap();

        self.get(model_info)
    }

    fn get(&self, model_info: &ModelInfo) -> Self::M {
        let mapper = Mapper::new();

        //cache(
            TensorflowModel::new(
                model_info.clone(),
                Engine::new(),
                mapper
            )
        //)
    }

    fn get_latest(&self, model_info: &ModelInfo) -> Result<ModelInfo, Error> {
        Ok(get_latest_model_info(model_info)?)
    }
}

#[cfg(test)]
mod tests {
    extern crate test;

    use super::*;
    use model::tensorflow::model::{Mapper as MapperTrait};
    use test::Bencher;

    #[test]
    fn test_map_action_to_policy_output_idx_pawn_a7_up() {
        let action = Action::Move(Square::new('a', 7), Direction::Up);
        let idx = map_action_to_policy_output_idx(&action);

        assert_eq!(0, idx);
    }

    #[test]
    fn test_map_action_to_policy_output_idx_pawn_a1_up() {
        let action = Action::Move(Square::new('a', 1), Direction::Up);
        let idx = map_action_to_policy_output_idx(&action);

        assert_eq!(48, idx);
    }

    #[test]
    fn test_map_action_to_policy_output_idx_pawn_h7_up() {
        let action = Action::Move(Square::new('h', 7), Direction::Up);
        let idx = map_action_to_policy_output_idx(&action);

        assert_eq!(7, idx);
    }

    #[test]
    fn test_map_action_to_policy_output_idx_pawn_h1_up() {
        let action = Action::Move(Square::new('h', 1), Direction::Up);
        let idx = map_action_to_policy_output_idx(&action);

        assert_eq!(NUM_UP_MOVES - 1, idx);
    }

    #[test]
    fn test_map_action_to_policy_output_idx_pawn_a8_right() {
        let action = Action::Move(Square::new('a', 8), Direction::Right);
        let idx = map_action_to_policy_output_idx(&action);

        assert_eq!(NUM_UP_MOVES + 0, idx);
    }

    #[test]
    fn test_map_action_to_policy_output_idx_pawn_a1_right() {
        let action = Action::Move(Square::new('a', 1), Direction::Right);
        let idx = map_action_to_policy_output_idx(&action);

        assert_eq!(NUM_UP_MOVES + 49, idx);
    }

    #[test]
    fn test_map_action_to_policy_output_idx_pawn_g8_right() {
        let action = Action::Move(Square::new('g', 8), Direction::Right);
        let idx = map_action_to_policy_output_idx(&action);

        assert_eq!(NUM_UP_MOVES + 6, idx);
    }

    #[test]
    fn test_map_action_to_policy_output_idx_pawn_g1_right() {
        let action = Action::Move(Square::new('g', 1), Direction::Right);
        let idx = map_action_to_policy_output_idx(&action);

        assert_eq!(NUM_UP_MOVES + NUM_RIGHT_MOVES - 1, idx);
    }

    #[test]
    fn test_map_action_to_policy_output_idx_pawn_a8_down() {
        let action = Action::Move(Square::new('a', 8), Direction::Down);
        let idx = map_action_to_policy_output_idx(&action);

        assert_eq!(NUM_UP_MOVES + NUM_RIGHT_MOVES + 0, idx);
    }

    #[test]
    fn test_map_action_to_policy_output_idx_pawn_a2_down() {
        let action = Action::Move(Square::new('a', 2), Direction::Down);
        let idx = map_action_to_policy_output_idx(&action);

        assert_eq!(NUM_UP_MOVES + NUM_RIGHT_MOVES + 48, idx);
    }

    #[test]
    fn test_map_action_to_policy_output_idx_pawn_h8_down() {
        let action = Action::Move(Square::new('h', 8), Direction::Down);
        let idx = map_action_to_policy_output_idx(&action);

        assert_eq!(NUM_UP_MOVES + NUM_RIGHT_MOVES + 7, idx);
    }

    #[test]
    fn test_map_action_to_policy_output_idx_pawn_h2_down() {
        let action = Action::Move(Square::new('h', 2), Direction::Down);
        let idx = map_action_to_policy_output_idx(&action);

        assert_eq!(NUM_UP_MOVES + NUM_RIGHT_MOVES + NUM_DOWN_MOVES - 1, idx);
    }

    #[test]
    fn test_map_action_to_policy_output_idx_pawn_b8_left() {
        let action = Action::Move(Square::new('b', 8), Direction::Left);
        let idx = map_action_to_policy_output_idx(&action);

        assert_eq!(NUM_UP_MOVES + NUM_RIGHT_MOVES + NUM_DOWN_MOVES + 0, idx);
    }

    #[test]
    fn test_map_action_to_policy_output_idx_pawn_b1_left() {
        let action = Action::Move(Square::new('b', 1), Direction::Left);
        let idx = map_action_to_policy_output_idx(&action);

        assert_eq!(NUM_UP_MOVES + NUM_RIGHT_MOVES + NUM_DOWN_MOVES + 49, idx);
    }

    #[test]
    fn test_map_action_to_policy_output_idx_pawn_h8_left() {
        let action = Action::Move(Square::new('h', 8), Direction::Left);
        let idx = map_action_to_policy_output_idx(&action);

        assert_eq!(NUM_UP_MOVES + NUM_RIGHT_MOVES + NUM_DOWN_MOVES + 6, idx);
    }

    #[test]
    fn test_map_action_to_policy_output_idx_pawn_h1_left() {
        let action = Action::Move(Square::new('h', 1), Direction::Left);
        let idx = map_action_to_policy_output_idx(&action);

        assert_eq!(NUM_UP_MOVES + NUM_RIGHT_MOVES + NUM_DOWN_MOVES + NUM_LEFT_MOVES - 1, idx);
    }

    #[test]
    fn test_map_action_to_policy_output_idx_pass() {
        let action = Action::Pass;
        let idx = map_action_to_policy_output_idx(&action);

        assert_eq!(NUM_UP_MOVES + NUM_RIGHT_MOVES + NUM_DOWN_MOVES + NUM_LEFT_MOVES, idx);
    }

    #[test]
    fn test_game_state_to_input_inverts() {
        let game_state: GameState = "
             1g
              +-----------------+
             8| r               |
             7|                 |
             6|     x     x     |
             5|                 |
             4|       E         |
             3|     x     x     |
             2|                 |
             1| R               |
              +-----------------+
                a b c d e f g h"
            .parse().unwrap();

        let mapper = Mapper::new();
        let game_state_to_input = mapper.game_state_to_input(&game_state);

        let game_state_inverted: GameState = "
             1s
              +-----------------+
             8|               r |
             7|                 |
             6|     x     x     |
             5|         e       |
             4|                 |
             3|     x     x     |
             2|                 |
             1|               R |
              +-----------------+
                a b c d e f g h"
            .parse().unwrap();

        let game_state_to_input_inverted = mapper.game_state_to_input(&game_state_inverted);

        assert_eq!(game_state_to_input, game_state_to_input_inverted);
    }

    const NUM_CHANNELS_PER_STEP: usize = 12;
    fn get_step_as_vec(input: &[f32], step: usize) -> Vec<f32> {
        let mut output = Vec::new();
        for offset in ((step - 1) * NUM_CHANNELS_PER_STEP)..(step * NUM_CHANNELS_PER_STEP) {
            output.extend(input.iter().skip(offset).step_by(INPUT_C).map(|i| *i));
        }

        output
    }

    #[test]
    fn test_game_state_to_input_steps() {
        let game_state_step1: GameState = "
             1g
              +-----------------+
             8| r               |
             7|                 |
             6|     x     x     |
             5|                 |
             4|       E         |
             3|     x     x     |
             2|                 |
             1| R               |
              +-----------------+
                a b c d e f g h"
            .parse().unwrap();
        
        let mapper = Mapper::new();
        let game_state_to_input_step1 = mapper.game_state_to_input(&game_state_step1);

        let game_state_step2 = game_state_step1.take_action(&Action::Move(Square::new('a', 1), Direction::Up));

        let game_state_step2_as_first_step: GameState = "
             1g
              +-----------------+
             8| r               |
             7|                 |
             6|     x     x     |
             5|                 |
             4|       E         |
             3|     x     x     |
             2| R               |
             1|                 |
              +-----------------+
                a b c d e f g h"
            .parse().unwrap();

        let game_state_to_input_step2_as_first_step = mapper.game_state_to_input(&game_state_step2_as_first_step);
        let game_state_to_input_step2 = mapper.game_state_to_input(&game_state_step2);

        assert_eq!(get_step_as_vec(&game_state_to_input_step1, 1), get_step_as_vec(&game_state_to_input_step2, 2));
        assert_eq!(get_step_as_vec(&game_state_to_input_step2_as_first_step, 1), get_step_as_vec(&game_state_to_input_step2, 1));

        let game_state_step3 = game_state_step2.take_action(&Action::Move(Square::new('a', 2), Direction::Up));

        let game_state_step3_as_first_step: GameState = "
             1g
              +-----------------+
             8| r               |
             7|                 |
             6|     x     x     |
             5|                 |
             4|       E         |
             3| R   x     x     |
             2|                 |
             1|                 |
              +-----------------+
                a b c d e f g h"
            .parse().unwrap();

        let game_state_to_input_step3_as_first_step = mapper.game_state_to_input(&game_state_step3_as_first_step);
        let game_state_to_input_step3 = mapper.game_state_to_input(&game_state_step3);

        assert_eq!(get_step_as_vec(&game_state_to_input_step1, 1), get_step_as_vec(&game_state_to_input_step3, 3));
        assert_eq!(get_step_as_vec(&game_state_to_input_step2, 1), get_step_as_vec(&game_state_to_input_step3, 2));
        assert_eq!(get_step_as_vec(&game_state_to_input_step3_as_first_step, 1), get_step_as_vec(&game_state_to_input_step3, 1));

        let game_state_step4 = game_state_step3.take_action(&Action::Move(Square::new('a', 3), Direction::Right));

        let game_state_step4_as_first_step: GameState = "
             1g
              +-----------------+
             8| r               |
             7|                 |
             6|     x     x     |
             5|                 |
             4|       E         |
             3|   R x     x     |
             2|                 |
             1|                 |
              +-----------------+
                a b c d e f g h"
            .parse().unwrap();

        let game_state_to_input_step4_as_first_step = mapper.game_state_to_input(&game_state_step4_as_first_step);
        let game_state_to_input_step4 = mapper.game_state_to_input(&game_state_step4);

        assert_eq!(get_step_as_vec(&game_state_to_input_step1, 1), get_step_as_vec(&game_state_to_input_step4, 4));
        assert_eq!(get_step_as_vec(&game_state_to_input_step2, 1), get_step_as_vec(&game_state_to_input_step4, 3));
        assert_eq!(get_step_as_vec(&game_state_to_input_step3, 1), get_step_as_vec(&game_state_to_input_step4, 2));
        assert_eq!(get_step_as_vec(&game_state_to_input_step4_as_first_step, 1), get_step_as_vec(&game_state_to_input_step4, 1));
    }

    #[bench]
    fn bench_game_state_to_input(b: &mut Bencher) {
        let mapper = Mapper::new();
        let game_state: GameState = "
             1s
              +-----------------+
             8| h c d m r d c h |
             7| r r r r e r r r |
             6|     x     x     |
             5|                 |
             4|                 |
             3|     x     x     |
             2| R R R R E R R R |
             1| H C D M R D C H |
              +-----------------+
                a b c d e f g h
            ".parse().unwrap();

        b.iter(|| mapper.game_state_to_input(&game_state));
    }
}
