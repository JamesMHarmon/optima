use half::f16;

use super::board::set_placement_board_bits;
use super::constants::PLACE_INPUT_SIZE as INPUT_SIZE;
use super::constants::{PLACE_INPUT_C as INPUT_C, PLACE_INPUT_H as INPUT_H};
use super::constants::{PLACE_INPUT_W as INPUT_W, PLACE_OUTPUT_SIZE as OUTPUT_SIZE, *};
use super::game_state::GameState;
use super::value::Value;
use super::PlaceTranspositionEntry;
use arimaa_engine::{Action, Piece};
use engine::value::Value as ValueTrait;
use model::analytics::{ActionWithPolicy, GameStateAnalysis};
use model::logits::update_logit_policies_to_softmax;
use model::node_metrics::NodeMetrics;
use tensorflow_model::{InputMap, Mode, PolicyMap};
use tensorflow_model::{TranspositionMap, ValueMap};

#[derive(Default)]
pub struct Mapper {}

impl Mapper {
    pub fn new() -> Self {
        Self::default()
    }
}

impl tensorflow_model::Dimension for Mapper {
    fn dimensions(&self) -> [u64; 3] {
        [INPUT_H as u64, INPUT_W as u64, INPUT_C as u64]
    }
}

impl InputMap<GameState> for Mapper {
    fn game_state_to_input(&self, game_state: &GameState, _mode: Mode) -> Vec<f16> {
        let mut input: Vec<f16> = vec![f16::ZERO; INPUT_SIZE];

        let is_p1_turn_to_move = game_state.is_p1_turn_to_move();
        let piece_board = game_state.get_piece_board();
        let player_piece_board = piece_board.get_player_piece_mask(is_p1_turn_to_move);

        for (i, piece) in [
            Piece::Elephant,
            Piece::Camel,
            Piece::Horse,
            Piece::Dog,
            Piece::Cat,
            Piece::Rabbit,
        ]
        .iter()
        .enumerate()
        {
            let piece_bits = piece_board.get_bits_by_piece_type(*piece);
            let offset = i;

            insert_input_channel_bits(&mut input, offset, piece_bits);

            let num_pieces_placed = (piece_bits & player_piece_board).count_ones();
            let num_pieces_to_place = match piece {
                Piece::Elephant | Piece::Camel => 1,
                Piece::Horse | Piece::Dog | Piece::Cat => 2,
                Piece::Rabbit => 8,
            };

            let offset = NUM_PIECE_TYPES + i;
            let num_piece_remaining =
                f16::from_f32(num_pieces_placed as f32 / num_pieces_to_place as f32);
            insert_input_channel_bit(&mut input, offset, num_piece_remaining);
        }

        let offset = NUM_PIECE_TYPES + NUM_PIECE_TYPES;
        let placement_bit = piece_board.get_placement_bit();
        insert_input_channel_bits(&mut input, offset, placement_bit);

        let offset = NUM_PIECE_TYPES + NUM_PIECE_TYPES + PLACEMENT_BIT_CHANNEL;
        let is_p1_turn_value = if is_p1_turn_to_move {
            f16::ZERO
        } else {
            f16::ONE
        };
        insert_input_channel_bit(&mut input, offset, is_p1_turn_value);

        input
    }
}

impl PolicyMap<GameState, Action> for Mapper {
    fn policy_metrics_to_expected_output(
        &self,
        _game_state: &GameState,
        policy_metrics: &NodeMetrics<Action>,
    ) -> Vec<f32> {
        let total_visits = policy_metrics.visits as f32 - 1.0;
        let inputs: Vec<f32> = vec![0f32; OUTPUT_SIZE];

        policy_metrics
            .children
            .iter()
            .fold(inputs, |mut r, (action, _w, visits)| {
                let policy_index = map_action_to_policy_output_idx(action);

                r[policy_index] = *visits as f32 / total_visits;
                r
            })
    }

    fn policy_to_valid_actions(
        &self,
        game_state: &GameState,
        policy_scores: &[f16],
    ) -> Vec<ActionWithPolicy<Action>> {
        let mut valid_actions_with_policies: Vec<_> = game_state
            .valid_actions()
            .into_iter()
            .map(|action| {
                let policy_index = map_action_to_policy_output_idx(&action);
                let policy_score = policy_scores[policy_index];

                ActionWithPolicy::new(action, policy_score.to_f32())
            })
            .collect();

        update_logit_policies_to_softmax(&mut valid_actions_with_policies);

        valid_actions_with_policies
    }
}

impl ValueMap<GameState, Value> for Mapper {
    fn map_value_to_value_output(&self, game_state: &GameState, value: &Value) -> f32 {
        let player_to_move = if game_state.is_p1_turn_to_move() {
            1
        } else {
            2
        };
        let val = value.get_value_for_player(player_to_move);
        (val * 2.0) - 1.0
    }

    fn map_value_output_to_value(&self, game_state: &GameState, value_output: f32) -> Value {
        let curr_val = (value_output + 1.0) / 2.0;
        let opp_val = 1.0 - curr_val;
        if game_state.is_p1_turn_to_move() {
            [curr_val, opp_val].into()
        } else {
            [opp_val, curr_val].into()
        }
    }
}

impl TranspositionMap<GameState, Action, Value, PlaceTranspositionEntry> for Mapper {
    fn get_transposition_key(&self, game_state: &GameState) -> u64 {
        game_state.get_transposition_hash()
    }

    fn map_output_to_transposition_entry<I: Iterator<Item = f16>>(
        &self,
        _game_state: &GameState,
        policy_scores: I,
        value: f16,
        moves_left: f32,
    ) -> PlaceTranspositionEntry {
        let mut policy_metrics = [f16::ZERO; OUTPUT_SIZE];

        for (i, score) in policy_scores.enumerate() {
            policy_metrics[i] = score;
        }

        PlaceTranspositionEntry::new(policy_metrics, value, moves_left)
    }

    fn map_transposition_entry_to_analysis(
        &self,
        game_state: &GameState,
        transposition_entry: &PlaceTranspositionEntry,
    ) -> GameStateAnalysis<Action, Value> {
        GameStateAnalysis::new(
            self.map_value_output_to_value(game_state, transposition_entry.value().to_f32()),
            self.policy_to_valid_actions(game_state, transposition_entry.policy_metrics()),
            transposition_entry.moves_left(),
        )
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
            Piece::Rabbit => 5,
        },
        _ => panic!("Action not expected"),
    }
}

fn insert_input_channel_bit(input: &mut [f16], offset: usize, value: f16) {
    for board_idx in 0..PLACE_BOARD_SIZE {
        let cell_idx = board_idx * INPUT_C + offset;
        input[cell_idx] = value;
    }
}

fn insert_input_channel_bits(input: &mut [f16], offset: usize, bits: u64) {
    set_placement_board_bits(input, offset, bits);
}
