use common::MovesLeftPropagatedValue;
use half::f16;
use mcts::{map_moves_left_to_one_hot, moves_left_expected_value};
use once_cell::sync::OnceCell;
use std::collections::HashMap;
use std::convert::TryInto;
use tinyvec::ArrayVec;

use super::{
    GameState, Predictions, TranspositionEntry, Value, constants::*, set_board_bits_invertable,
};
use arimaa_engine::{Action, MoveDirection, Path, Piece, PushPullDirection, Square};
use engine::value::Value as ValueTrait;
use model::analytics::{ActionWithPolicy, GameStateAnalysis};
use model::logits::update_logit_policies_to_softmax;
use model::node_metrics::NodeMetrics;
use tensorflow_model::{InputMap, Mode, PredictionsMap, TranspositionMap};

/*
    Layers:
    In:
    6 curr piece boards
    6 opp piece boards
    3 current step
    1 banned pieces board
    1 phase (play or setup)
    1 trap squares

    Out:
    40 directional boards (subtract irrelevant squares)
    12 push pull boards
    1 pass bit
    1 setup squares (16 logits)
*/
#[derive(Clone, Default)]
pub struct Mapper {}

impl Mapper {
    pub fn new() -> Self {
        Self {}
    }

    fn metrics_to_policy_output(
        &self,
        game_state: &GameState,
        policy_metrics: &NodeMetrics<Action, Predictions, MovesLeftPropagatedValue>,
    ) -> Vec<f32> {
        let move_map = static_sparse_piece_move_map().as_slice();
        let push_pull_map = static_sparse_push_pull_map().as_slice();
        let total_visits = policy_metrics.visits as f32 - 1.0;
        let invert = !game_state.is_p1_turn_to_move();
        let inputs: Vec<f32> = vec![-1f32; OUTPUT_SIZE];

        assert_eq!(
            policy_metrics.visits - 1,
            policy_metrics
                .children
                .iter()
                .map(|m| m.visits())
                .sum::<usize>(),
            "Sum of policy metrics should match policy metrics visits. {:?}",
            policy_metrics
        );

        policy_metrics.children.iter().fold(inputs, |mut r, m| {
            // Policy scores are in the perspective of player 1. That means that if we are p2, we need to flip the actions as if we were looking
            // at the board from the perspective of player 1, but with the pieces inverted.
            let policy_index = if invert {
                map_action_to_policy_output_idx(move_map, push_pull_map, &m.action().rotate())
            } else {
                map_action_to_policy_output_idx(move_map, push_pull_map, m.action())
            };

            assert!(
                (r[policy_index] - -1f32).abs() <= f32::EPSILON,
                "Policy value already exists {:?}",
                m.action()
            );

            r[policy_index] = m.visits() as f32 / total_visits;

            r
        })
    }

    pub fn policy_to_valid_actions(
        &self,
        game_state: &GameState,
        policy_scores: &[f16],
    ) -> Vec<ActionWithPolicy<Action>> {
        let invert = !game_state.is_p1_turn_to_move();
        let move_map = static_sparse_piece_move_map().as_slice();
        let push_pull_map = static_sparse_push_pull_map().as_slice();

        let mut valid_actions_with_policies: Vec<_> = game_state
            .valid_actions()
            .into_iter()
            .map(|action| {
                // Policy scores coming from the model are always from the perspective of player 1.
                // This means that if we are p2, we need to flip the actions coming back and translate them
                // to be actions in the p2 perspective.
                let policy_index = if invert {
                    map_action_to_policy_output_idx(move_map, push_pull_map, &action.rotate())
                } else {
                    map_action_to_policy_output_idx(move_map, push_pull_map, &action)
                };

                let policy_score = policy_scores[policy_index];

                ActionWithPolicy::new(action, policy_score)
            })
            .collect();

        update_logit_policies_to_softmax(&mut valid_actions_with_policies);

        valid_actions_with_policies
    }

    fn metrics_to_value_output(&self, game_state: &GameState, value: &Value) -> Vec<f32> {
        let player_to_move = game_state.player_to_move();
        let val = value.get_value_for_player(player_to_move);
        vec![(val * 2.0) - 1.0]
    }

    fn map_value_output_to_value(&self, game_state: &GameState, value_output: f16) -> Value {
        let curr_val = (f16::to_f32(value_output) + 1.0) / 2.0;
        let opp_val = 1.0 - curr_val;
        if game_state.is_p1_turn_to_move() {
            [curr_val, opp_val].into()
        } else {
            [opp_val, curr_val].into()
        }
    }
}

impl tensorflow_model::Dimension for Mapper {
    fn dimensions(&self) -> [u64; 3] {
        [INPUT_H as u64, INPUT_W as u64, INPUT_C as u64]
    }
}

impl InputMap for Mapper {
    type State = GameState;

    fn game_state_to_input(&self, game_state: &GameState, input: &mut [f16], _mode: Mode) {
        set_board_state_squares(input, game_state);

        set_step_num_squares(input, game_state);

        set_banned_piece_squares(input, game_state);

        set_phase_squares(input, game_state);

        set_trap_squares(input);
    }
}

impl PredictionsMap for Mapper {
    type State = GameState;
    type Action = Action;
    type Predictions = Predictions;
    type PropagatedValues = MovesLeftPropagatedValue;

    fn to_output(
        &self,
        game_state: &Self::State,
        targets: Self::Predictions,
        node_metrics: &NodeMetrics<Self::Action, Self::Predictions, Self::PropagatedValues>,
    ) -> std::collections::HashMap<String, Vec<f32>> {
        let policy_output = self.metrics_to_policy_output(game_state, node_metrics);
        let value_output = self.metrics_to_value_output(game_state, targets.value());

        let move_number = game_state.get_move_number() as f32;
        let moves_left = (targets.game_length() - move_number + 1.0).max(1.0);
        let moves_left_one_hot = map_moves_left_to_one_hot(moves_left, MOVES_LEFT_SIZE);

        /*
            Validate the outputs to ensure values are in their appropriate ranges.
        */
        let sum_of_policy = policy_output.iter().filter(|&&x| x >= 0.0).sum::<f32>();
        assert!(
            f32::abs(sum_of_policy - 1.0) <= f32::EPSILON * policy_output.len() as f32,
            "Policy output should sum to 1.0 but actual sum is {}",
            sum_of_policy
        );

        for policy in policy_output.iter() {
            assert!(
                (0.0..=1.0).contains(policy) || *policy == -1.0,
                "Policy output should be in range 0.0-1.0 but was {}",
                policy
            );
        }

        for value in value_output.iter() {
            assert!(
                (-1.0..=1.0).contains(value),
                "Value output should be in range -1.0-1.0 but was {}",
                value
            );
        }

        assert_eq!(policy_output.len(), OUTPUT_SIZE);
        assert_eq!(value_output.len(), 1);
        assert_eq!(moves_left_one_hot.len(), MOVES_LEFT_SIZE);

        [
            ("policy", policy_output),
            ("value", value_output),
            ("moves_left", moves_left_one_hot),
        ]
        .into_iter()
        .map(|(key, value)| (key.to_string(), value))
        .collect()
    }
}

impl TranspositionMap for Mapper {
    type State = GameState;
    type Action = Action;
    type TranspositionEntry = TranspositionEntry;
    type Predictions = Predictions;

    fn get_transposition_key(&self, game_state: &GameState) -> u64 {
        game_state.get_transposition_hash() ^ game_state.get_banned_piece_mask()
    }

    fn map_output_to_transposition_entry(
        &self,
        game_state: &GameState,
        outputs: HashMap<String, &[f16]>,
    ) -> TranspositionEntry {
        let policy_scores = *outputs
            .get("policy_head")
            .expect("Policy scores not found in output");

        let policy_metrics = policy_scores
            .try_into()
            .expect("Slice does not match length of array");

        let value = outputs
            .get("value_head")
            .expect("Value not found in output")[0];

        let moves_left_vals = outputs
            .get("moves_left_head")
            .expect("Moves left not found in output");

        let moves_left = moves_left_expected_value(moves_left_vals.iter().map(|x| x.to_f32()));

        let game_length = (game_state.get_move_number() as f32 + moves_left - 1.0).max(1.0);

        TranspositionEntry::new(policy_metrics, value, game_length)
    }

    fn map_transposition_entry_to_analysis(
        &self,
        game_state: &GameState,
        transposition_entry: &TranspositionEntry,
    ) -> GameStateAnalysis<Action, Predictions> {
        let predictions = Predictions::new(
            self.map_value_output_to_value(game_state, transposition_entry.value()),
            transposition_entry.game_length(),
        );

        GameStateAnalysis::new(
            self.policy_to_valid_actions(game_state, transposition_entry.policy_metrics()),
            predictions,
        )
    }
}

fn set_board_state_squares(input: &mut [f16], game_state: &GameState) {
    let is_p1_turn_to_move = game_state.is_p1_turn_to_move();
    let invert = !is_p1_turn_to_move;

    let piece_board = game_state.get_piece_board();

    for (j, player) in [is_p1_turn_to_move, !is_p1_turn_to_move].iter().enumerate() {
        let player_offset = j * NUM_PIECE_TYPES;

        for (piece_offset, piece) in [
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
            let piece_bits = piece_board.get_bits_for_piece(*piece, *player);

            let channel_idx = player_offset + piece_offset;
            set_board_bits_invertable(input, channel_idx, piece_bits, invert);
        }
    }
}

fn set_banned_piece_squares(input: &mut [f16], game_state: &GameState) {
    let is_p1_turn_to_move = game_state.is_p1_turn_to_move();
    let invert = !is_p1_turn_to_move;
    let bits = game_state.get_banned_piece_mask();
    set_board_bits_invertable(input, BANNED_PIECES_CHANNEL_IDX, bits, invert);
}

fn set_step_num_squares(input: &mut [f16], game_state: &GameState) {
    let current_step = game_state.step();

    // Current step is base 0. However we start from 1 since the first step doesn't have a corresponding channel since 0 0 0 represents the first step.
    for step_num in 1..=current_step {
        let step_num_channel_idx = STEP_NUM_CHANNEL_IDX + step_num - 1;

        set_all_bits_for_channel(input, step_num_channel_idx);
    }
}

fn set_phase_squares(input: &mut [f16], game_state: &GameState) {
    if !game_state.is_play_phase() {
        set_all_bits_for_channel(input, PHASE_CHANNEL_IDX);
    }
}

fn set_all_bits_for_channel(input: &mut [f16], channel_idx: usize) {
    for board_idx in 0..BOARD_SIZE {
        let input_idx = board_idx * INPUT_C + channel_idx;
        input[input_idx] = f16::ONE;
    }
}

fn set_trap_squares(input: &mut [f16]) {
    input[INPUT_C * 18 + TRAP_CHANNEL_IDX] = f16::ONE;
    input[INPUT_C * 21 + TRAP_CHANNEL_IDX] = f16::ONE;
    input[INPUT_C * 42 + TRAP_CHANNEL_IDX] = f16::ONE;
    input[INPUT_C * 45 + TRAP_CHANNEL_IDX] = f16::ONE;
}

fn map_action_to_policy_output_idx(
    move_map: &[u16],
    push_pull_map: &[u16],
    action: &Action,
) -> usize {
    match action {
        Action::Move(square, path) => {
            move_map[map_square_path_to_sparse_idx(*square, *path)] as usize
        }
        Action::PushPull(square, dir) => {
            NUM_PIECE_MOVES
                + push_pull_map[map_square_push_pull_to_sparse_idx(square, dir)] as usize
        }
        Action::Pass => NUM_PIECE_MOVES + NUM_PUSH_PULL_MOVES,
        Action::Place(square) => {
            let idx = square.get_index() - (64 - PLACE_MOVES);
            NUM_PIECE_MOVES + NUM_PUSH_PULL_MOVES + PASS_MOVES + idx
        }
    }
}

fn map_square_path_to_sparse_idx(square: Square, path: Path) -> usize {
    (((square.get_index() as u16) << 8) ^ (path.as_u8() as u16)) as usize
}

fn is_direction_valid_from_square(square: Square, move_direction: MoveDirection) -> bool {
    let mut shifted_sq = square;
    for direction in move_direction.directions() {
        if !shifted_sq.can_shift_in_direction(direction) {
            return false;
        }

        shifted_sq = shifted_sq.shift_in_direction(direction);
    }

    true
}

fn get_valid_paths_with_squares() -> Vec<(Square, MoveDirection, ArrayVec<[Path; 6]>)> {
    arimaa_engine::get_path_permutations_by_move_direction()
        .iter()
        .flat_map(|move_direction| {
            (0..64).map(move |idx| (Square::from_index(idx), move_direction))
        })
        .filter(|(square, (move_direction, _))| {
            is_direction_valid_from_square(*square, *move_direction)
        })
        .map(|(square, (move_direction, paths))| (square, *move_direction, *paths))
        .collect()
}

fn static_sparse_piece_move_map() -> &'static Vec<u16> {
    static INSTANCE: OnceCell<Vec<u16>> = OnceCell::new();
    INSTANCE.get_or_init(|| {
        let valid_moves = get_valid_paths_with_squares();
        assert_eq!(valid_moves.len(), NUM_PIECE_MOVES);

        let mut sparse_map = vec![u16::MAX; 16361];

        for (idx, (square, _, paths)) in valid_moves.into_iter().enumerate() {
            for path in paths {
                let sparse_idx = map_square_path_to_sparse_idx(square, path);

                assert_eq!(
                    sparse_map[sparse_idx],
                    u16::MAX,
                    "Collision found while filling sparse map"
                );

                sparse_map[sparse_idx] = idx as u16;
            }
        }

        sparse_map
    })
}

fn map_square_push_pull_to_sparse_idx(square: &Square, dir: &PushPullDirection) -> usize {
    (((square.get_index() as u16) << 4) ^ (dir.as_u8() as u16)) as usize
}

fn get_valid_push_pulls_with_squares() -> Vec<(Square, PushPullDirection)> {
    PushPullDirection::directions()
        .into_iter()
        .flat_map(|push_pull_direction| {
            (0..64).map(move |idx| (Square::from_index(idx), push_pull_direction))
        })
        .filter(|(square, push_pull_direction)| push_pull_direction.is_valid_from_square(*square))
        .collect()
}

fn static_sparse_push_pull_map() -> &'static Vec<u16> {
    static INSTANCE: OnceCell<Vec<u16>> = OnceCell::new();
    INSTANCE.get_or_init(|| {
        let valid_moves = get_valid_push_pulls_with_squares();
        assert_eq!(
            valid_moves.len(),
            NUM_PUSH_PULL_MOVES,
            "The number of valid push pull moves should be correct."
        );

        let mut sparse_map = vec![u16::MAX; 1019];

        for (idx, (square, dir)) in valid_moves.iter().enumerate() {
            let sparse_idx = map_square_push_pull_to_sparse_idx(square, dir);

            assert_eq!(
                sparse_map[sparse_idx],
                u16::MAX,
                "Collision found while filling sparse map"
            );

            sparse_map[sparse_idx] = idx as u16;
        }

        sparse_map
    })
}

#[allow(clippy::float_cmp)]
#[cfg(test)]
mod tests {
    use super::*;
    use arimaa_engine::take_actions;
    use engine::GameState as GameStateTrait;
    use itertools::Itertools;
    use model::EdgeMetrics;

    fn map_action_to_policy_output_idx(action: &Action) -> usize {
        let move_map = static_sparse_piece_move_map().as_slice();
        let push_pull_map = static_sparse_push_pull_map().as_slice();
        super::map_action_to_policy_output_idx(move_map, push_pull_map, action)
    }

    fn game_state_to_input_fn(game_state: &GameState, mode: Mode) -> Vec<f16> {
        let mapper = Mapper::new();
        let mut input = vec![f16::ZERO; INPUT_SIZE];
        mapper.game_state_to_input(game_state, &mut input, mode);
        input
    }

    #[test]
    fn test_map_action_to_policy_output_idx_piece_move_a7n() {
        let action = "a7n".parse().unwrap();
        let idx = map_action_to_policy_output_idx(&action);

        assert_eq!(0, idx);
    }

    #[test]
    fn test_map_action_to_policy_output_idx_piece_move_b7n() {
        let action = "b7n".parse().unwrap();
        let idx = map_action_to_policy_output_idx(&action);

        assert_eq!(1, idx);
    }

    #[test]
    fn test_map_action_to_policy_output_idx_piece_move_h1n() {
        let action = "h1n".parse().unwrap();
        let idx = map_action_to_policy_output_idx(&action);

        assert_eq!(55, idx);
    }

    #[test]
    fn test_map_action_to_policy_output_idx_piece_move_a8e() {
        let action = "a8e".parse().unwrap();
        let idx = map_action_to_policy_output_idx(&action);

        assert_eq!(56, idx);
    }

    #[test]
    fn test_map_action_to_policy_output_idx_piece_move_h1wwww() {
        let action = "h1wwww".parse().unwrap();
        let idx = map_action_to_policy_output_idx(&action);

        assert_eq!(1659, idx);
    }

    #[test]
    fn test_map_action_to_policy_output_idx_piece_move_all() {
        let mut num_asserts = 0;

        for (i, (square, _, paths)) in get_valid_paths_with_squares().into_iter().enumerate() {
            for path in paths {
                let action = Action::Move(square, path);
                let idx = map_action_to_policy_output_idx(&action);
                assert_eq!(i, idx);
            }
            num_asserts += 1;
        }

        assert_eq!(num_asserts, NUM_PIECE_MOVES)
    }

    #[test]
    fn test_valid_paths_is_in_the_correct_order_of_move_dirs() {
        for (dir_a, dir_b) in get_valid_paths_with_squares()
            .into_iter()
            .map(|(_, dir, _)| dir)
            .dedup()
            .zip(MoveDirection::move_directions())
        {
            assert_eq!(dir_a, dir_b);
        }
    }

    #[test]
    fn test_map_action_to_policy_output_idx_piece_move_action_strings() {
        let actions = [
            "a7n", "b7n", "c7n", "d7n", "e7n", "f7n", "g7n", "h7n", "a6n", "b6n", "c6n", "d6n",
            "e6n", "f6n", "g6n", "h6n", "a5n", "b5n", "c5n", "d5n", "e5n", "f5n", "g5n", "h5n",
            "a4n", "b4n", "c4n", "d4n", "e4n", "f4n", "g4n", "h4n", "a3n", "b3n", "c3n", "d3n",
            "e3n", "f3n", "g3n", "h3n", "a2n", "b2n", "c2n", "d2n", "e2n", "f2n", "g2n", "h2n",
            "a1n", "b1n", "c1n", "d1n", "e1n", "f1n", "g1n", "h1n", "a8e", "b8e", "c8e", "d8e",
            "e8e", "f8e", "g8e", "a7e", "b7e", "c7e", "d7e", "e7e", "f7e", "g7e", "a6e", "b6e",
            "c6e", "d6e", "e6e", "f6e", "g6e", "a5e", "b5e", "c5e", "d5e", "e5e", "f5e", "g5e",
            "a4e", "b4e", "c4e", "d4e", "e4e", "f4e", "g4e", "a3e", "b3e", "c3e", "d3e", "e3e",
            "f3e", "g3e", "a2e", "b2e", "c2e", "d2e", "e2e", "f2e", "g2e", "a1e", "b1e", "c1e",
            "d1e", "e1e", "f1e", "g1e", "a8s", "b8s", "c8s", "d8s", "e8s", "f8s", "g8s", "h8s",
            "a7s", "b7s", "c7s", "d7s", "e7s", "f7s", "g7s", "h7s", "a6s", "b6s", "c6s", "d6s",
            "e6s", "f6s", "g6s", "h6s", "a5s", "b5s", "c5s", "d5s", "e5s", "f5s", "g5s", "h5s",
            "a4s", "b4s", "c4s", "d4s", "e4s", "f4s", "g4s", "h4s", "a3s", "b3s", "c3s", "d3s",
            "e3s", "f3s", "g3s", "h3s", "a2s", "b2s", "c2s", "d2s", "e2s", "f2s", "g2s", "h2s",
            "b8w", "c8w", "d8w", "e8w", "f8w", "g8w", "h8w", "b7w", "c7w", "d7w", "e7w", "f7w",
            "g7w", "h7w", "b6w", "c6w", "d6w", "e6w", "f6w", "g6w", "h6w", "b5w", "c5w", "d5w",
            "e5w", "f5w", "g5w", "h5w", "b4w", "c4w", "d4w", "e4w", "f4w", "g4w", "h4w", "b3w",
            "c3w", "d3w", "e3w", "f3w", "g3w", "h3w", "b2w", "c2w", "d2w", "e2w", "f2w", "g2w",
            "h2w", "b1w", "c1w", "d1w", "e1w", "f1w", "g1w", "h1w", "a6nn", "b6nn", "c6nn", "d6nn",
            "e6nn", "f6nn", "g6nn", "h6nn", "a5nn", "b5nn", "c5nn", "d5nn", "e5nn", "f5nn", "g5nn",
            "h5nn", "a4nn", "b4nn", "c4nn", "d4nn", "e4nn", "f4nn", "g4nn", "h4nn", "a3nn", "b3nn",
            "c3nn", "d3nn", "e3nn", "f3nn", "g3nn", "h3nn", "a2nn", "b2nn", "c2nn", "d2nn", "e2nn",
            "f2nn", "g2nn", "h2nn", "a1nn", "b1nn", "c1nn", "d1nn", "e1nn", "f1nn", "g1nn", "h1nn",
            "a7ne", "b7ne", "c7ne", "d7ne", "e7ne", "f7ne", "g7ne", "a6ne", "b6ne", "c6ne", "d6ne",
            "e6ne", "f6ne", "g6ne", "a5ne", "b5ne", "c5ne", "d5ne", "e5ne", "f5ne", "g5ne", "a4ne",
            "b4ne", "c4ne", "d4ne", "e4ne", "f4ne", "g4ne", "a3ne", "b3ne", "c3ne", "d3ne", "e3ne",
            "f3ne", "g3ne", "a2ne", "b2ne", "c2ne", "d2ne", "e2ne", "f2ne", "g2ne", "a1ne", "b1ne",
            "c1ne", "d1ne", "e1ne", "f1ne", "g1ne", "b7nw", "c7nw", "d7nw", "e7nw", "f7nw", "g7nw",
            "h7nw", "b6nw", "c6nw", "d6nw", "e6nw", "f6nw", "g6nw", "h6nw", "b5nw", "c5nw", "d5nw",
            "e5nw", "f5nw", "g5nw", "h5nw", "b4nw", "c4nw", "d4nw", "e4nw", "f4nw", "g4nw", "h4nw",
            "b3nw", "c3nw", "d3nw", "e3nw", "f3nw", "g3nw", "h3nw", "b2nw", "c2nw", "d2nw", "e2nw",
            "f2nw", "g2nw", "h2nw", "b1nw", "c1nw", "d1nw", "e1nw", "f1nw", "g1nw", "h1nw", "a8ee",
            "b8ee", "c8ee", "d8ee", "e8ee", "f8ee", "a7ee", "b7ee", "c7ee", "d7ee", "e7ee", "f7ee",
            "a6ee", "b6ee", "c6ee", "d6ee", "e6ee", "f6ee", "a5ee", "b5ee", "c5ee", "d5ee", "e5ee",
            "f5ee", "a4ee", "b4ee", "c4ee", "d4ee", "e4ee", "f4ee", "a3ee", "b3ee", "c3ee", "d3ee",
            "e3ee", "f3ee", "a2ee", "b2ee", "c2ee", "d2ee", "e2ee", "f2ee", "a1ee", "b1ee", "c1ee",
            "d1ee", "e1ee", "f1ee", "a8es", "b8es", "c8es", "d8es", "e8es", "f8es", "g8es", "a7es",
            "b7es", "c7es", "d7es", "e7es", "f7es", "g7es", "a6es", "b6es", "c6es", "d6es", "e6es",
            "f6es", "g6es", "a5es", "b5es", "c5es", "d5es", "e5es", "f5es", "g5es", "a4es", "b4es",
            "c4es", "d4es", "e4es", "f4es", "g4es", "a3es", "b3es", "c3es", "d3es", "e3es", "f3es",
            "g3es", "a2es", "b2es", "c2es", "d2es", "e2es", "f2es", "g2es", "a8ss", "b8ss", "c8ss",
            "d8ss", "e8ss", "f8ss", "g8ss", "h8ss", "a7ss", "b7ss", "c7ss", "d7ss", "e7ss", "f7ss",
            "g7ss", "h7ss", "a6ss", "b6ss", "c6ss", "d6ss", "e6ss", "f6ss", "g6ss", "h6ss", "a5ss",
            "b5ss", "c5ss", "d5ss", "e5ss", "f5ss", "g5ss", "h5ss", "a4ss", "b4ss", "c4ss", "d4ss",
            "e4ss", "f4ss", "g4ss", "h4ss", "a3ss", "b3ss", "c3ss", "d3ss", "e3ss", "f3ss", "g3ss",
            "h3ss", "b8sw", "c8sw", "d8sw", "e8sw", "f8sw", "g8sw", "h8sw", "b7sw", "c7sw", "d7sw",
            "e7sw", "f7sw", "g7sw", "h7sw", "b6sw", "c6sw", "d6sw", "e6sw", "f6sw", "g6sw", "h6sw",
            "b5sw", "c5sw", "d5sw", "e5sw", "f5sw", "g5sw", "h5sw", "b4sw", "c4sw", "d4sw", "e4sw",
            "f4sw", "g4sw", "h4sw", "b3sw", "c3sw", "d3sw", "e3sw", "f3sw", "g3sw", "h3sw", "b2sw",
            "c2sw", "d2sw", "e2sw", "f2sw", "g2sw", "h2sw", "c8ww", "d8ww", "e8ww", "f8ww", "g8ww",
            "h8ww", "c7ww", "d7ww", "e7ww", "f7ww", "g7ww", "h7ww", "c6ww", "d6ww", "e6ww", "f6ww",
            "g6ww", "h6ww", "c5ww", "d5ww", "e5ww", "f5ww", "g5ww", "h5ww", "c4ww", "d4ww", "e4ww",
            "f4ww", "g4ww", "h4ww", "c3ww", "d3ww", "e3ww", "f3ww", "g3ww", "h3ww", "c2ww", "d2ww",
            "e2ww", "f2ww", "g2ww", "h2ww", "c1ww", "d1ww", "e1ww", "f1ww", "g1ww", "h1ww",
            "a5nnn", "b5nnn", "c5nnn", "d5nnn", "e5nnn", "f5nnn", "g5nnn", "h5nnn", "a4nnn",
            "b4nnn", "c4nnn", "d4nnn", "e4nnn", "f4nnn", "g4nnn", "h4nnn", "a3nnn", "b3nnn",
            "c3nnn", "d3nnn", "e3nnn", "f3nnn", "g3nnn", "h3nnn", "a2nnn", "b2nnn", "c2nnn",
            "d2nnn", "e2nnn", "f2nnn", "g2nnn", "h2nnn", "a1nnn", "b1nnn", "c1nnn", "d1nnn",
            "e1nnn", "f1nnn", "g1nnn", "h1nnn", "a6nne", "b6nne", "c6nne", "d6nne", "e6nne",
            "f6nne", "g6nne", "a5nne", "b5nne", "c5nne", "d5nne", "e5nne", "f5nne", "g5nne",
            "a4nne", "b4nne", "c4nne", "d4nne", "e4nne", "f4nne", "g4nne", "a3nne", "b3nne",
            "c3nne", "d3nne", "e3nne", "f3nne", "g3nne", "a2nne", "b2nne", "c2nne", "d2nne",
            "e2nne", "f2nne", "g2nne", "a1nne", "b1nne", "c1nne", "d1nne", "e1nne", "f1nne",
            "g1nne", "b6nnw", "c6nnw", "d6nnw", "e6nnw", "f6nnw", "g6nnw", "h6nnw", "b5nnw",
            "c5nnw", "d5nnw", "e5nnw", "f5nnw", "g5nnw", "h5nnw", "b4nnw", "c4nnw", "d4nnw",
            "e4nnw", "f4nnw", "g4nnw", "h4nnw", "b3nnw", "c3nnw", "d3nnw", "e3nnw", "f3nnw",
            "g3nnw", "h3nnw", "b2nnw", "c2nnw", "d2nnw", "e2nnw", "f2nnw", "g2nnw", "h2nnw",
            "b1nnw", "c1nnw", "d1nnw", "e1nnw", "f1nnw", "g1nnw", "h1nnw", "a7nee", "b7nee",
            "c7nee", "d7nee", "e7nee", "f7nee", "a6nee", "b6nee", "c6nee", "d6nee", "e6nee",
            "f6nee", "a5nee", "b5nee", "c5nee", "d5nee", "e5nee", "f5nee", "a4nee", "b4nee",
            "c4nee", "d4nee", "e4nee", "f4nee", "a3nee", "b3nee", "c3nee", "d3nee", "e3nee",
            "f3nee", "a2nee", "b2nee", "c2nee", "d2nee", "e2nee", "f2nee", "a1nee", "b1nee",
            "c1nee", "d1nee", "e1nee", "f1nee", "c7nww", "d7nww", "e7nww", "f7nww", "g7nww",
            "h7nww", "c6nww", "d6nww", "e6nww", "f6nww", "g6nww", "h6nww", "c5nww", "d5nww",
            "e5nww", "f5nww", "g5nww", "h5nww", "c4nww", "d4nww", "e4nww", "f4nww", "g4nww",
            "h4nww", "c3nww", "d3nww", "e3nww", "f3nww", "g3nww", "h3nww", "c2nww", "d2nww",
            "e2nww", "f2nww", "g2nww", "h2nww", "c1nww", "d1nww", "e1nww", "f1nww", "g1nww",
            "h1nww", "a8eee", "b8eee", "c8eee", "d8eee", "e8eee", "a7eee", "b7eee", "c7eee",
            "d7eee", "e7eee", "a6eee", "b6eee", "c6eee", "d6eee", "e6eee", "a5eee", "b5eee",
            "c5eee", "d5eee", "e5eee", "a4eee", "b4eee", "c4eee", "d4eee", "e4eee", "a3eee",
            "b3eee", "c3eee", "d3eee", "e3eee", "a2eee", "b2eee", "c2eee", "d2eee", "e2eee",
            "a1eee", "b1eee", "c1eee", "d1eee", "e1eee", "a8ees", "b8ees", "c8ees", "d8ees",
            "e8ees", "f8ees", "a7ees", "b7ees", "c7ees", "d7ees", "e7ees", "f7ees", "a6ees",
            "b6ees", "c6ees", "d6ees", "e6ees", "f6ees", "a5ees", "b5ees", "c5ees", "d5ees",
            "e5ees", "f5ees", "a4ees", "b4ees", "c4ees", "d4ees", "e4ees", "f4ees", "a3ees",
            "b3ees", "c3ees", "d3ees", "e3ees", "f3ees", "a2ees", "b2ees", "c2ees", "d2ees",
            "e2ees", "f2ees", "a8ess", "b8ess", "c8ess", "d8ess", "e8ess", "f8ess", "g8ess",
            "a7ess", "b7ess", "c7ess", "d7ess", "e7ess", "f7ess", "g7ess", "a6ess", "b6ess",
            "c6ess", "d6ess", "e6ess", "f6ess", "g6ess", "a5ess", "b5ess", "c5ess", "d5ess",
            "e5ess", "f5ess", "g5ess", "a4ess", "b4ess", "c4ess", "d4ess", "e4ess", "f4ess",
            "g4ess", "a3ess", "b3ess", "c3ess", "d3ess", "e3ess", "f3ess", "g3ess", "a8sss",
            "b8sss", "c8sss", "d8sss", "e8sss", "f8sss", "g8sss", "h8sss", "a7sss", "b7sss",
            "c7sss", "d7sss", "e7sss", "f7sss", "g7sss", "h7sss", "a6sss", "b6sss", "c6sss",
            "d6sss", "e6sss", "f6sss", "g6sss", "h6sss", "a5sss", "b5sss", "c5sss", "d5sss",
            "e5sss", "f5sss", "g5sss", "h5sss", "a4sss", "b4sss", "c4sss", "d4sss", "e4sss",
            "f4sss", "g4sss", "h4sss", "b8ssw", "c8ssw", "d8ssw", "e8ssw", "f8ssw", "g8ssw",
            "h8ssw", "b7ssw", "c7ssw", "d7ssw", "e7ssw", "f7ssw", "g7ssw", "h7ssw", "b6ssw",
            "c6ssw", "d6ssw", "e6ssw", "f6ssw", "g6ssw", "h6ssw", "b5ssw", "c5ssw", "d5ssw",
            "e5ssw", "f5ssw", "g5ssw", "h5ssw", "b4ssw", "c4ssw", "d4ssw", "e4ssw", "f4ssw",
            "g4ssw", "h4ssw", "b3ssw", "c3ssw", "d3ssw", "e3ssw", "f3ssw", "g3ssw", "h3ssw",
            "c8sww", "d8sww", "e8sww", "f8sww", "g8sww", "h8sww", "c7sww", "d7sww", "e7sww",
            "f7sww", "g7sww", "h7sww", "c6sww", "d6sww", "e6sww", "f6sww", "g6sww", "h6sww",
            "c5sww", "d5sww", "e5sww", "f5sww", "g5sww", "h5sww", "c4sww", "d4sww", "e4sww",
            "f4sww", "g4sww", "h4sww", "c3sww", "d3sww", "e3sww", "f3sww", "g3sww", "h3sww",
            "c2sww", "d2sww", "e2sww", "f2sww", "g2sww", "h2sww", "d8www", "e8www", "f8www",
            "g8www", "h8www", "d7www", "e7www", "f7www", "g7www", "h7www", "d6www", "e6www",
            "f6www", "g6www", "h6www", "d5www", "e5www", "f5www", "g5www", "h5www", "d4www",
            "e4www", "f4www", "g4www", "h4www", "d3www", "e3www", "f3www", "g3www", "h3www",
            "d2www", "e2www", "f2www", "g2www", "h2www", "d1www", "e1www", "f1www", "g1www",
            "h1www", "a4nnnn", "b4nnnn", "c4nnnn", "d4nnnn", "e4nnnn", "f4nnnn", "g4nnnn",
            "h4nnnn", "a3nnnn", "b3nnnn", "c3nnnn", "d3nnnn", "e3nnnn", "f3nnnn", "g3nnnn",
            "h3nnnn", "a2nnnn", "b2nnnn", "c2nnnn", "d2nnnn", "e2nnnn", "f2nnnn", "g2nnnn",
            "h2nnnn", "a1nnnn", "b1nnnn", "c1nnnn", "d1nnnn", "e1nnnn", "f1nnnn", "g1nnnn",
            "h1nnnn", "a5nnne", "b5nnne", "c5nnne", "d5nnne", "e5nnne", "f5nnne", "g5nnne",
            "a4nnne", "b4nnne", "c4nnne", "d4nnne", "e4nnne", "f4nnne", "g4nnne", "a3nnne",
            "b3nnne", "c3nnne", "d3nnne", "e3nnne", "f3nnne", "g3nnne", "a2nnne", "b2nnne",
            "c2nnne", "d2nnne", "e2nnne", "f2nnne", "g2nnne", "a1nnne", "b1nnne", "c1nnne",
            "d1nnne", "e1nnne", "f1nnne", "g1nnne", "b5nnnw", "c5nnnw", "d5nnnw", "e5nnnw",
            "f5nnnw", "g5nnnw", "h5nnnw", "b4nnnw", "c4nnnw", "d4nnnw", "e4nnnw", "f4nnnw",
            "g4nnnw", "h4nnnw", "b3nnnw", "c3nnnw", "d3nnnw", "e3nnnw", "f3nnnw", "g3nnnw",
            "h3nnnw", "b2nnnw", "c2nnnw", "d2nnnw", "e2nnnw", "f2nnnw", "g2nnnw", "h2nnnw",
            "b1nnnw", "c1nnnw", "d1nnnw", "e1nnnw", "f1nnnw", "g1nnnw", "h1nnnw", "a6nnee",
            "b6nnee", "c6nnee", "d6nnee", "e6nnee", "f6nnee", "a5nnee", "b5nnee", "c5nnee",
            "d5nnee", "e5nnee", "f5nnee", "a4nnee", "b4nnee", "c4nnee", "d4nnee", "e4nnee",
            "f4nnee", "a3nnee", "b3nnee", "c3nnee", "d3nnee", "e3nnee", "f3nnee", "a2nnee",
            "b2nnee", "c2nnee", "d2nnee", "e2nnee", "f2nnee", "a1nnee", "b1nnee", "c1nnee",
            "d1nnee", "e1nnee", "f1nnee", "c6nnww", "d6nnww", "e6nnww", "f6nnww", "g6nnww",
            "h6nnww", "c5nnww", "d5nnww", "e5nnww", "f5nnww", "g5nnww", "h5nnww", "c4nnww",
            "d4nnww", "e4nnww", "f4nnww", "g4nnww", "h4nnww", "c3nnww", "d3nnww", "e3nnww",
            "f3nnww", "g3nnww", "h3nnww", "c2nnww", "d2nnww", "e2nnww", "f2nnww", "g2nnww",
            "h2nnww", "c1nnww", "d1nnww", "e1nnww", "f1nnww", "g1nnww", "h1nnww", "a7neee",
            "b7neee", "c7neee", "d7neee", "e7neee", "a6neee", "b6neee", "c6neee", "d6neee",
            "e6neee", "a5neee", "b5neee", "c5neee", "d5neee", "e5neee", "a4neee", "b4neee",
            "c4neee", "d4neee", "e4neee", "a3neee", "b3neee", "c3neee", "d3neee", "e3neee",
            "a2neee", "b2neee", "c2neee", "d2neee", "e2neee", "a1neee", "b1neee", "c1neee",
            "d1neee", "e1neee", "d7nwww", "e7nwww", "f7nwww", "g7nwww", "h7nwww", "d6nwww",
            "e6nwww", "f6nwww", "g6nwww", "h6nwww", "d5nwww", "e5nwww", "f5nwww", "g5nwww",
            "h5nwww", "d4nwww", "e4nwww", "f4nwww", "g4nwww", "h4nwww", "d3nwww", "e3nwww",
            "f3nwww", "g3nwww", "h3nwww", "d2nwww", "e2nwww", "f2nwww", "g2nwww", "h2nwww",
            "d1nwww", "e1nwww", "f1nwww", "g1nwww", "h1nwww", "a8eeee", "b8eeee", "c8eeee",
            "d8eeee", "a7eeee", "b7eeee", "c7eeee", "d7eeee", "a6eeee", "b6eeee", "c6eeee",
            "d6eeee", "a5eeee", "b5eeee", "c5eeee", "d5eeee", "a4eeee", "b4eeee", "c4eeee",
            "d4eeee", "a3eeee", "b3eeee", "c3eeee", "d3eeee", "a2eeee", "b2eeee", "c2eeee",
            "d2eeee", "a1eeee", "b1eeee", "c1eeee", "d1eeee", "a8eees", "b8eees", "c8eees",
            "d8eees", "e8eees", "a7eees", "b7eees", "c7eees", "d7eees", "e7eees", "a6eees",
            "b6eees", "c6eees", "d6eees", "e6eees", "a5eees", "b5eees", "c5eees", "d5eees",
            "e5eees", "a4eees", "b4eees", "c4eees", "d4eees", "e4eees", "a3eees", "b3eees",
            "c3eees", "d3eees", "e3eees", "a2eees", "b2eees", "c2eees", "d2eees", "e2eees",
            "a8eess", "b8eess", "c8eess", "d8eess", "e8eess", "f8eess", "a7eess", "b7eess",
            "c7eess", "d7eess", "e7eess", "f7eess", "a6eess", "b6eess", "c6eess", "d6eess",
            "e6eess", "f6eess", "a5eess", "b5eess", "c5eess", "d5eess", "e5eess", "f5eess",
            "a4eess", "b4eess", "c4eess", "d4eess", "e4eess", "f4eess", "a3eess", "b3eess",
            "c3eess", "d3eess", "e3eess", "f3eess", "a8esss", "b8esss", "c8esss", "d8esss",
            "e8esss", "f8esss", "g8esss", "a7esss", "b7esss", "c7esss", "d7esss", "e7esss",
            "f7esss", "g7esss", "a6esss", "b6esss", "c6esss", "d6esss", "e6esss", "f6esss",
            "g6esss", "a5esss", "b5esss", "c5esss", "d5esss", "e5esss", "f5esss", "g5esss",
            "a4esss", "b4esss", "c4esss", "d4esss", "e4esss", "f4esss", "g4esss", "a8ssss",
            "b8ssss", "c8ssss", "d8ssss", "e8ssss", "f8ssss", "g8ssss", "h8ssss", "a7ssss",
            "b7ssss", "c7ssss", "d7ssss", "e7ssss", "f7ssss", "g7ssss", "h7ssss", "a6ssss",
            "b6ssss", "c6ssss", "d6ssss", "e6ssss", "f6ssss", "g6ssss", "h6ssss", "a5ssss",
            "b5ssss", "c5ssss", "d5ssss", "e5ssss", "f5ssss", "g5ssss", "h5ssss", "b8sssw",
            "c8sssw", "d8sssw", "e8sssw", "f8sssw", "g8sssw", "h8sssw", "b7sssw", "c7sssw",
            "d7sssw", "e7sssw", "f7sssw", "g7sssw", "h7sssw", "b6sssw", "c6sssw", "d6sssw",
            "e6sssw", "f6sssw", "g6sssw", "h6sssw", "b5sssw", "c5sssw", "d5sssw", "e5sssw",
            "f5sssw", "g5sssw", "h5sssw", "b4sssw", "c4sssw", "d4sssw", "e4sssw", "f4sssw",
            "g4sssw", "h4sssw", "c8ssww", "d8ssww", "e8ssww", "f8ssww", "g8ssww", "h8ssww",
            "c7ssww", "d7ssww", "e7ssww", "f7ssww", "g7ssww", "h7ssww", "c6ssww", "d6ssww",
            "e6ssww", "f6ssww", "g6ssww", "h6ssww", "c5ssww", "d5ssww", "e5ssww", "f5ssww",
            "g5ssww", "h5ssww", "c4ssww", "d4ssww", "e4ssww", "f4ssww", "g4ssww", "h4ssww",
            "c3ssww", "d3ssww", "e3ssww", "f3ssww", "g3ssww", "h3ssww", "d8swww", "e8swww",
            "f8swww", "g8swww", "h8swww", "d7swww", "e7swww", "f7swww", "g7swww", "h7swww",
            "d6swww", "e6swww", "f6swww", "g6swww", "h6swww", "d5swww", "e5swww", "f5swww",
            "g5swww", "h5swww", "d4swww", "e4swww", "f4swww", "g4swww", "h4swww", "d3swww",
            "e3swww", "f3swww", "g3swww", "h3swww", "d2swww", "e2swww", "f2swww", "g2swww",
            "h2swww", "e8wwww", "f8wwww", "g8wwww", "h8wwww", "e7wwww", "f7wwww", "g7wwww",
            "h7wwww", "e6wwww", "f6wwww", "g6wwww", "h6wwww", "e5wwww", "f5wwww", "g5wwww",
            "h5wwww", "e4wwww", "f4wwww", "g4wwww", "h4wwww", "e3wwww", "f3wwww", "g3wwww",
            "h3wwww", "e2wwww", "f2wwww", "g2wwww", "h2wwww", "e1wwww", "f1wwww", "g1wwww",
            "h1wwww",
        ];

        for (i, action) in actions.iter().enumerate() {
            let idx = map_action_to_policy_output_idx(&action.parse().unwrap());
            assert_eq!(i, idx);
        }
    }

    #[test]
    fn test_map_action_to_policy_output_idx_push_pull_pa7nn() {
        let action = "pa7nn".parse().unwrap();
        let idx = map_action_to_policy_output_idx(&action);

        assert_eq!(NUM_PIECE_MOVES, idx);
    }

    #[test]
    fn test_map_action_to_policy_output_idx_push_pull_pb7nn() {
        let action = "pb7nn".parse().unwrap();
        let idx = map_action_to_policy_output_idx(&action);

        assert_eq!(NUM_PIECE_MOVES + 1, idx);
    }

    #[test]
    fn test_map_action_to_policy_output_idx_push_pull_pg1ww() {
        let action = "pg1ww".parse().unwrap();
        let idx = map_action_to_policy_output_idx(&action);

        assert_eq!(NUM_PIECE_MOVES + 583, idx);
    }

    #[test]
    fn test_map_action_to_policy_output_idx_push_pull_all() {
        let mut num_asserts = 0;

        for (i, (square, dir)) in get_valid_push_pulls_with_squares().into_iter().enumerate() {
            let action = Action::PushPull(square, dir);
            let idx = map_action_to_policy_output_idx(&action);
            assert_eq!(NUM_PIECE_MOVES + i, idx);
            num_asserts += 1;
        }

        assert_eq!(num_asserts, NUM_PUSH_PULL_MOVES)
    }

    #[test]
    fn test_map_action_to_policy_output_idx_push_pull_action_strings() {
        let actions = [
            "pa7nn", "pb7nn", "pc7nn", "pd7nn", "pe7nn", "pf7nn", "pg7nn", "ph7nn", "pa6nn",
            "pb6nn", "pc6nn", "pd6nn", "pe6nn", "pf6nn", "pg6nn", "ph6nn", "pa5nn", "pb5nn",
            "pc5nn", "pd5nn", "pe5nn", "pf5nn", "pg5nn", "ph5nn", "pa4nn", "pb4nn", "pc4nn",
            "pd4nn", "pe4nn", "pf4nn", "pg4nn", "ph4nn", "pa3nn", "pb3nn", "pc3nn", "pd3nn",
            "pe3nn", "pf3nn", "pg3nn", "ph3nn", "pa2nn", "pb2nn", "pc2nn", "pd2nn", "pe2nn",
            "pf2nn", "pg2nn", "ph2nn", "pb7ne", "pc7ne", "pd7ne", "pe7ne", "pf7ne", "pg7ne",
            "ph7ne", "pb6ne", "pc6ne", "pd6ne", "pe6ne", "pf6ne", "pg6ne", "ph6ne", "pb5ne",
            "pc5ne", "pd5ne", "pe5ne", "pf5ne", "pg5ne", "ph5ne", "pb4ne", "pc4ne", "pd4ne",
            "pe4ne", "pf4ne", "pg4ne", "ph4ne", "pb3ne", "pc3ne", "pd3ne", "pe3ne", "pf3ne",
            "pg3ne", "ph3ne", "pb2ne", "pc2ne", "pd2ne", "pe2ne", "pf2ne", "pg2ne", "ph2ne",
            "pb1ne", "pc1ne", "pd1ne", "pe1ne", "pf1ne", "pg1ne", "ph1ne", "pa7nw", "pb7nw",
            "pc7nw", "pd7nw", "pe7nw", "pf7nw", "pg7nw", "pa6nw", "pb6nw", "pc6nw", "pd6nw",
            "pe6nw", "pf6nw", "pg6nw", "pa5nw", "pb5nw", "pc5nw", "pd5nw", "pe5nw", "pf5nw",
            "pg5nw", "pa4nw", "pb4nw", "pc4nw", "pd4nw", "pe4nw", "pf4nw", "pg4nw", "pa3nw",
            "pb3nw", "pc3nw", "pd3nw", "pe3nw", "pf3nw", "pg3nw", "pa2nw", "pb2nw", "pc2nw",
            "pd2nw", "pe2nw", "pf2nw", "pg2nw", "pa1nw", "pb1nw", "pc1nw", "pd1nw", "pe1nw",
            "pf1nw", "pg1nw", "pa8en", "pb8en", "pc8en", "pd8en", "pe8en", "pf8en", "pg8en",
            "pa7en", "pb7en", "pc7en", "pd7en", "pe7en", "pf7en", "pg7en", "pa6en", "pb6en",
            "pc6en", "pd6en", "pe6en", "pf6en", "pg6en", "pa5en", "pb5en", "pc5en", "pd5en",
            "pe5en", "pf5en", "pg5en", "pa4en", "pb4en", "pc4en", "pd4en", "pe4en", "pf4en",
            "pg4en", "pa3en", "pb3en", "pc3en", "pd3en", "pe3en", "pf3en", "pg3en", "pa2en",
            "pb2en", "pc2en", "pd2en", "pe2en", "pf2en", "pg2en", "pb8ee", "pc8ee", "pd8ee",
            "pe8ee", "pf8ee", "pg8ee", "pb7ee", "pc7ee", "pd7ee", "pe7ee", "pf7ee", "pg7ee",
            "pb6ee", "pc6ee", "pd6ee", "pe6ee", "pf6ee", "pg6ee", "pb5ee", "pc5ee", "pd5ee",
            "pe5ee", "pf5ee", "pg5ee", "pb4ee", "pc4ee", "pd4ee", "pe4ee", "pf4ee", "pg4ee",
            "pb3ee", "pc3ee", "pd3ee", "pe3ee", "pf3ee", "pg3ee", "pb2ee", "pc2ee", "pd2ee",
            "pe2ee", "pf2ee", "pg2ee", "pb1ee", "pc1ee", "pd1ee", "pe1ee", "pf1ee", "pg1ee",
            "pa7es", "pb7es", "pc7es", "pd7es", "pe7es", "pf7es", "pg7es", "pa6es", "pb6es",
            "pc6es", "pd6es", "pe6es", "pf6es", "pg6es", "pa5es", "pb5es", "pc5es", "pd5es",
            "pe5es", "pf5es", "pg5es", "pa4es", "pb4es", "pc4es", "pd4es", "pe4es", "pf4es",
            "pg4es", "pa3es", "pb3es", "pc3es", "pd3es", "pe3es", "pf3es", "pg3es", "pa2es",
            "pb2es", "pc2es", "pd2es", "pe2es", "pf2es", "pg2es", "pa1es", "pb1es", "pc1es",
            "pd1es", "pe1es", "pf1es", "pg1es", "pb8se", "pc8se", "pd8se", "pe8se", "pf8se",
            "pg8se", "ph8se", "pb7se", "pc7se", "pd7se", "pe7se", "pf7se", "pg7se", "ph7se",
            "pb6se", "pc6se", "pd6se", "pe6se", "pf6se", "pg6se", "ph6se", "pb5se", "pc5se",
            "pd5se", "pe5se", "pf5se", "pg5se", "ph5se", "pb4se", "pc4se", "pd4se", "pe4se",
            "pf4se", "pg4se", "ph4se", "pb3se", "pc3se", "pd3se", "pe3se", "pf3se", "pg3se",
            "ph3se", "pb2se", "pc2se", "pd2se", "pe2se", "pf2se", "pg2se", "ph2se", "pa7ss",
            "pb7ss", "pc7ss", "pd7ss", "pe7ss", "pf7ss", "pg7ss", "ph7ss", "pa6ss", "pb6ss",
            "pc6ss", "pd6ss", "pe6ss", "pf6ss", "pg6ss", "ph6ss", "pa5ss", "pb5ss", "pc5ss",
            "pd5ss", "pe5ss", "pf5ss", "pg5ss", "ph5ss", "pa4ss", "pb4ss", "pc4ss", "pd4ss",
            "pe4ss", "pf4ss", "pg4ss", "ph4ss", "pa3ss", "pb3ss", "pc3ss", "pd3ss", "pe3ss",
            "pf3ss", "pg3ss", "ph3ss", "pa2ss", "pb2ss", "pc2ss", "pd2ss", "pe2ss", "pf2ss",
            "pg2ss", "ph2ss", "pa8sw", "pb8sw", "pc8sw", "pd8sw", "pe8sw", "pf8sw", "pg8sw",
            "pa7sw", "pb7sw", "pc7sw", "pd7sw", "pe7sw", "pf7sw", "pg7sw", "pa6sw", "pb6sw",
            "pc6sw", "pd6sw", "pe6sw", "pf6sw", "pg6sw", "pa5sw", "pb5sw", "pc5sw", "pd5sw",
            "pe5sw", "pf5sw", "pg5sw", "pa4sw", "pb4sw", "pc4sw", "pd4sw", "pe4sw", "pf4sw",
            "pg4sw", "pa3sw", "pb3sw", "pc3sw", "pd3sw", "pe3sw", "pf3sw", "pg3sw", "pa2sw",
            "pb2sw", "pc2sw", "pd2sw", "pe2sw", "pf2sw", "pg2sw", "pb8wn", "pc8wn", "pd8wn",
            "pe8wn", "pf8wn", "pg8wn", "ph8wn", "pb7wn", "pc7wn", "pd7wn", "pe7wn", "pf7wn",
            "pg7wn", "ph7wn", "pb6wn", "pc6wn", "pd6wn", "pe6wn", "pf6wn", "pg6wn", "ph6wn",
            "pb5wn", "pc5wn", "pd5wn", "pe5wn", "pf5wn", "pg5wn", "ph5wn", "pb4wn", "pc4wn",
            "pd4wn", "pe4wn", "pf4wn", "pg4wn", "ph4wn", "pb3wn", "pc3wn", "pd3wn", "pe3wn",
            "pf3wn", "pg3wn", "ph3wn", "pb2wn", "pc2wn", "pd2wn", "pe2wn", "pf2wn", "pg2wn",
            "ph2wn", "pb7ws", "pc7ws", "pd7ws", "pe7ws", "pf7ws", "pg7ws", "ph7ws", "pb6ws",
            "pc6ws", "pd6ws", "pe6ws", "pf6ws", "pg6ws", "ph6ws", "pb5ws", "pc5ws", "pd5ws",
            "pe5ws", "pf5ws", "pg5ws", "ph5ws", "pb4ws", "pc4ws", "pd4ws", "pe4ws", "pf4ws",
            "pg4ws", "ph4ws", "pb3ws", "pc3ws", "pd3ws", "pe3ws", "pf3ws", "pg3ws", "ph3ws",
            "pb2ws", "pc2ws", "pd2ws", "pe2ws", "pf2ws", "pg2ws", "ph2ws", "pb1ws", "pc1ws",
            "pd1ws", "pe1ws", "pf1ws", "pg1ws", "ph1ws", "pb8ww", "pc8ww", "pd8ww", "pe8ww",
            "pf8ww", "pg8ww", "pb7ww", "pc7ww", "pd7ww", "pe7ww", "pf7ww", "pg7ww", "pb6ww",
            "pc6ww", "pd6ww", "pe6ww", "pf6ww", "pg6ww", "pb5ww", "pc5ww", "pd5ww", "pe5ww",
            "pf5ww", "pg5ww", "pb4ww", "pc4ww", "pd4ww", "pe4ww", "pf4ww", "pg4ww", "pb3ww",
            "pc3ww", "pd3ww", "pe3ww", "pf3ww", "pg3ww", "pb2ww", "pc2ww", "pd2ww", "pe2ww",
            "pf2ww", "pg2ww", "pb1ww", "pc1ww", "pd1ww", "pe1ww", "pf1ww", "pg1ww",
        ];

        for (i, action) in actions.iter().enumerate() {
            let idx = map_action_to_policy_output_idx(&action.parse().unwrap());
            assert_eq!(NUM_PIECE_MOVES + i, idx);
        }
    }

    #[test]
    fn test_map_action_to_policy_output_idx_pass() {
        let action = Action::Pass;
        let idx = map_action_to_policy_output_idx(&action);

        assert_eq!(NUM_PIECE_MOVES + NUM_PUSH_PULL_MOVES, idx);
    }

    #[test]
    fn test_map_action_to_policy_output_idx_place_a2() {
        let action = "a2".parse().unwrap();
        let idx = map_action_to_policy_output_idx(&action);

        assert_eq!(NUM_PIECE_MOVES + NUM_PUSH_PULL_MOVES + PASS_MOVES, idx);
    }

    #[test]
    fn test_map_action_to_policy_output_idx_place_h2() {
        let action = "h2".parse().unwrap();
        let idx = map_action_to_policy_output_idx(&action);

        assert_eq!(NUM_PIECE_MOVES + NUM_PUSH_PULL_MOVES + PASS_MOVES + 7, idx);
    }

    #[test]
    fn test_map_action_to_policy_output_idx_place_a1() {
        let action = "a1".parse().unwrap();
        let idx = map_action_to_policy_output_idx(&action);

        assert_eq!(NUM_PIECE_MOVES + NUM_PUSH_PULL_MOVES + PASS_MOVES + 8, idx);
    }

    #[test]
    fn test_map_action_to_policy_output_idx_place_h1() {
        let action = "h1".parse().unwrap();
        let idx = map_action_to_policy_output_idx(&action);

        assert_eq!(NUM_PIECE_MOVES + NUM_PUSH_PULL_MOVES + PASS_MOVES + 15, idx);
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
            .parse()
            .unwrap();

        let game_state_to_input = game_state_to_input_fn(&game_state, Mode::Train);

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
            .parse()
            .unwrap();

        let game_state_to_input_inverted =
            game_state_to_input_fn(&game_state_inverted, Mode::Train);

        assert_eq!(game_state_to_input, game_state_to_input_inverted);
    }

    fn get_channel_as_vec(input: &[f16], channel_idx: usize) -> Vec<f32> {
        input
            .iter()
            .skip(channel_idx)
            .step_by(INPUT_C)
            .copied()
            .map(f16::to_f32)
            .collect()
    }

    #[test]
    fn test_game_state_to_input_step_num() {
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
            .parse()
            .unwrap();

        let assert_steps_set = |input: &Vec<f16>, first: bool, second: bool, third: bool| {
            let expected_step_channel_set =
                std::iter::repeat_n(1.0, BOARD_SIZE).collect::<Vec<_>>();
            let expected_step_channel_not_set =
                std::iter::repeat_n(0.0, BOARD_SIZE).collect::<Vec<_>>();

            let actual_step_channel_1 = get_channel_as_vec(input, STEP_NUM_CHANNEL_IDX);
            let actual_step_channel_2 = get_channel_as_vec(input, STEP_NUM_CHANNEL_IDX + 1);
            let actual_step_channel_3 = get_channel_as_vec(input, STEP_NUM_CHANNEL_IDX + 2);

            assert_eq!(
                &actual_step_channel_1,
                if first {
                    &expected_step_channel_set
                } else {
                    &expected_step_channel_not_set
                }
            );
            assert_eq!(
                &actual_step_channel_2,
                if second {
                    &expected_step_channel_set
                } else {
                    &expected_step_channel_not_set
                }
            );
            assert_eq!(
                &actual_step_channel_3,
                if third {
                    &expected_step_channel_set
                } else {
                    &expected_step_channel_not_set
                }
            );
        };

        let input = game_state_to_input_fn(&game_state, Mode::Train);

        assert_steps_set(&input, false, false, false);

        let game_state = game_state.take_action(&"a1n".parse().unwrap());
        let input = game_state_to_input_fn(&game_state, Mode::Train);

        assert_steps_set(&input, true, false, false);

        let game_state = game_state.take_action(&"a2n".parse().unwrap());
        let input = game_state_to_input_fn(&game_state, Mode::Train);

        assert_steps_set(&input, true, true, false);

        let game_state = game_state.take_action(&"a3n".parse().unwrap());
        let input = game_state_to_input_fn(&game_state, Mode::Train);

        assert_steps_set(&input, true, true, true);

        let game_state = game_state.take_action(&"a8s".parse().unwrap());
        let input = game_state_to_input_fn(&game_state, Mode::Train);

        assert_steps_set(&input, false, false, false);
    }

    fn string_to_vec(str: &str) -> Vec<f32> {
        regex::Regex::new(r"([0-9.]+)")
            .unwrap()
            .captures_iter(str)
            .map(|c| c[1].parse().unwrap())
            .collect()
    }

    fn count_set_bits(input: &[f16]) -> usize {
        input
            .iter()
            .copied()
            .map(f16::to_f32)
            .filter(|v| *v == 1.0)
            .count()
    }

    #[test]
    fn test_game_state_to_input_traps() {
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
            .parse()
            .unwrap();

        let input = game_state_to_input_fn(&game_state, Mode::Train);
        let expected_channel_traps = string_to_vec(
            "
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 1 0 0 1 0 0
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 1 0 0 1 0 0
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0",
        );

        let actual_trap_channel = get_channel_as_vec(&input, TRAP_CHANNEL_IDX);

        assert_eq!(actual_trap_channel, expected_channel_traps);
        assert_eq!(count_set_bits(&input), 7);
    }

    #[test]
    fn test_game_state_to_input_elephants() {
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
            .parse()
            .unwrap();

        let input = game_state_to_input_fn(&game_state, Mode::Train);
        let expected_elephants: Vec<_> = string_to_vec(
            "
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 0 1 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0",
        );

        let actual = get_channel_as_vec(&input, 0);

        assert_eq!(actual, expected_elephants);
        assert_eq!(count_set_bits(&input), 7);
    }

    #[test]
    fn test_game_state_to_input_rabbits_step_1() {
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
            .parse()
            .unwrap();

        let input = game_state_to_input_fn(&game_state, Mode::Train);
        let expected: Vec<_> = string_to_vec(
            "
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0
            1 0 0 0 0 0 0 0",
        );

        let actual = get_channel_as_vec(&input, 5);

        assert_eq!(actual, expected);
        assert_eq!(count_set_bits(&input), 7);
    }

    #[test]
    fn test_game_state_to_input_opp_rabbits() {
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
            .parse()
            .unwrap();

        let input = game_state_to_input_fn(&game_state, Mode::Train);
        let expected: Vec<_> = string_to_vec(
            "
                1 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0",
        );

        let actual = get_channel_as_vec(&input, 11);

        assert_eq!(actual, expected);
        assert_eq!(count_set_bits(&input), 7);
    }

    #[test]
    fn test_game_state_to_input_rabbits_step_2() {
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
            .parse()
            .unwrap();

        let game_state = game_state.take_action(&"a1n".parse().unwrap());

        let input = game_state_to_input_fn(&game_state, Mode::Train);
        let expected: Vec<_> = string_to_vec(
            "
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0
            1 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0",
        );

        let actual = get_channel_as_vec(&input, 5);
        assert_eq!(actual, expected);

        assert_eq!(count_set_bits(&input), 3 + 64 + 1 + 4); // pieces, step, banned pieces, traps
    }

    #[test]
    #[allow(clippy::identity_op)]
    fn test_game_state_to_input_as_silver() {
        let game_state: GameState = "
             1s
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
            .parse()
            .unwrap();

        let input = game_state_to_input_fn(&game_state, Mode::Train);
        let expected: Vec<_> = string_to_vec(
            "
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 1",
        );

        let actual = get_channel_as_vec(&input, 5);
        assert_eq!(actual, expected);

        let expected: Vec<_> = string_to_vec(
            "
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 0 0 1 0 0 0
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0",
        );

        let actual = get_channel_as_vec(&input, 6);
        assert_eq!(actual, expected);

        let expected: Vec<_> = string_to_vec(
            "
            0 0 0 0 0 0 0 1
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0",
        );

        let actual = get_channel_as_vec(&input, 11);
        assert_eq!(actual, expected);

        assert_eq!(count_set_bits(&input), 3 + 0 + 0 + 4); // pieces, step, banned pieces, traps
    }

    #[test]
    fn test_game_state_to_input_banned_piece_as_silver() {
        let game_state: GameState = "
             1s
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
            .parse()
            .unwrap();

        let game_state = game_state.take_action(&"a8s".parse().unwrap());
        let input = game_state_to_input_fn(&game_state, Mode::Train);

        let expected: Vec<_> = string_to_vec(
            "
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 1
            0 0 0 0 0 0 0 0",
        );

        let actual = get_channel_as_vec(&input, BANNED_PIECES_CHANNEL_IDX);
        assert_eq!(actual, expected);

        assert_eq!(count_set_bits(&input), 3 + 64 + 1 + 4); // pieces, step, banned pieces, traps
    }

    #[test]
    #[allow(clippy::identity_op)]
    fn test_game_state_to_input_rabbits_step_4() {
        let game_state: GameState = "
             1g
              +-----------------+
             8| r               |
             7|                 |
             6|     x r   x     |
             5|       E         |
             4|                 |
             3|     x     x     |
             2|                 |
             1| R               |
              +-----------------+
                a b c d e f g h"
            .parse()
            .unwrap();

        let expected: Vec<_> = string_to_vec(
            "
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 0 1 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0",
        );

        let input = game_state_to_input_fn(&game_state, Mode::Train);
        let actual = get_channel_as_vec(&input, 0);
        assert_eq!(actual, expected);

        let expected: Vec<_> = string_to_vec(
            "
            1 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 0 1 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0",
        );

        let actual = get_channel_as_vec(&input, 11);
        assert_eq!(actual, expected);

        let game_state = game_state.take_action(&"d6n".parse().unwrap());
        let input = game_state_to_input_fn(&game_state, Mode::Train);

        let expected: Vec<_> = string_to_vec(
            "
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 0 1 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0",
        );

        let actual = get_channel_as_vec(&input, 0);
        assert_eq!(actual, expected);

        let expected: Vec<_> = string_to_vec(
            "
            1 0 0 0 0 0 0 0
            0 0 0 1 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0",
        );

        let actual = get_channel_as_vec(&input, 11);
        assert_eq!(actual, expected);

        let game_state = game_state.take_action(&"d5n".parse().unwrap());
        let input = game_state_to_input_fn(&game_state, Mode::Train);

        let expected: Vec<_> = string_to_vec(
            "
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 0 1 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0",
        );

        let actual = get_channel_as_vec(&input, 0);
        assert_eq!(actual, expected);

        let expected: Vec<_> = string_to_vec(
            "
            1 0 0 0 0 0 0 0
            0 0 0 1 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0",
        );

        let actual = get_channel_as_vec(&input, 11);
        assert_eq!(actual, expected);

        let game_state = game_state.take_action(&"d6w".parse().unwrap());
        let input = game_state_to_input_fn(&game_state, Mode::Train);

        let expected: Vec<_> = string_to_vec(
            "
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0",
        );

        let actual = get_channel_as_vec(&input, 0);
        assert_eq!(actual, expected);

        let expected: Vec<_> = string_to_vec(
            "
            1 0 0 0 0 0 0 0
            0 0 0 1 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0",
        );

        let actual = get_channel_as_vec(&input, 11);
        assert_eq!(actual, expected);

        assert_eq!(count_set_bits(&input), 3 + 3 * 64 + 0 + 4); // pieces, step, banned pieces, traps
    }

    #[test]
    #[allow(clippy::identity_op)]
    fn test_game_state_to_input_phase_plane_filled_for_setup() {
        let game_state = GameState::initial();

        let input = game_state_to_input_fn(&game_state, Mode::Train);

        let expected: Vec<_> = string_to_vec(
            "
            1 1 1 1 1 1 1 1
            1 1 1 1 1 1 1 1
            1 1 1 1 1 1 1 1
            1 1 1 1 1 1 1 1
            1 1 1 1 1 1 1 1
            1 1 1 1 1 1 1 1
            1 1 1 1 1 1 1 1
            1 1 1 1 1 1 1 1",
        );

        let actual = get_channel_as_vec(&input, PHASE_CHANNEL_IDX);
        assert_eq!(actual, expected);

        assert_eq!(count_set_bits(&input), 0 + 0 + 0 + 4 + 64); // pieces, step, banned pieces, traps, phase
    }

    #[test]
    #[allow(clippy::identity_op)]
    fn test_game_state_to_input_phase_plane_full_last_setup_move() {
        let game_state = GameState::initial();
        let game_state =
            take_actions!(game_state => a1, b1, c1, d1, e1, f1, g1, h1, a7, b7, c7, d7, e7, f7, g7);

        let input = game_state_to_input_fn(&game_state, Mode::Train);

        let expected: Vec<_> = string_to_vec(
            "
            1 1 1 1 1 1 1 1
            1 1 1 1 1 1 1 1
            1 1 1 1 1 1 1 1
            1 1 1 1 1 1 1 1
            1 1 1 1 1 1 1 1
            1 1 1 1 1 1 1 1
            1 1 1 1 1 1 1 1
            1 1 1 1 1 1 1 1",
        );

        let actual = get_channel_as_vec(&input, PHASE_CHANNEL_IDX);
        assert_eq!(actual, expected);

        assert_eq!(count_set_bits(&input), 23 + 0 + 0 + 4 + 64); // pieces, step, banned pieces, traps, phase
    }

    #[test]
    #[allow(clippy::identity_op)]
    fn test_game_state_to_input_phase_plane_empty_for_play() {
        let game_state = GameState::initial();
        let game_state = take_actions!(game_state => a1, b1, c1, d1, e1, f1, g1, h1, a7, b7, c7, d7, e7, f7, g7, h7);

        let input = game_state_to_input_fn(&game_state, Mode::Train);

        let expected: Vec<_> = string_to_vec(
            "
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0",
        );

        let actual = get_channel_as_vec(&input, PHASE_CHANNEL_IDX);
        assert_eq!(actual, expected);

        assert_eq!(count_set_bits(&input), 32 + 0 + 0 + 4 + 0); // pieces, step, banned pieces, traps, phase
    }

    #[test]
    fn test_policy_metrics_to_expected_output() {
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
            .parse()
            .unwrap();

        let policy_metrics = NodeMetrics {
            visits: 11,
            predictions: Predictions::new(Value::new([0.0, 0.0]), 0.0),
            children: vec![
                EdgeMetrics::new(
                    "a7n".parse().unwrap(),
                    7,
                    MovesLeftPropagatedValue::new(0.0, 0.0),
                ),
                EdgeMetrics::new(
                    "pa7nn".parse().unwrap(),
                    2,
                    MovesLeftPropagatedValue::new(0.0, 0.0),
                ),
                EdgeMetrics::new(
                    "p".parse().unwrap(),
                    1,
                    MovesLeftPropagatedValue::new(0.0, 0.0),
                ),
            ],
        };

        let output = Mapper::new().metrics_to_policy_output(&game_state, &policy_metrics);
        let pass_idx = NUM_PIECE_MOVES + NUM_PUSH_PULL_MOVES;
        assert_eq!(output.len(), OUTPUT_SIZE);
        assert_eq!(
            output[0], 0.7,
            "Policy for a7n should be 7 visit / 10 visits"
        );
        assert_eq!(
            output[NUM_PIECE_MOVES], 0.2,
            "Policy for pa7nn should be 7 visit / 10 visits"
        );
        assert_eq!(
            output[pass_idx], 0.1,
            "Policy for pass should be 1 visit / 10 visits"
        );
    }

    #[test]
    fn test_policy_metrics_to_expected_output_silver() {
        let game_state: GameState = "
            1s
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
            .parse()
            .unwrap();

        let policy_metrics = NodeMetrics {
            visits: 11,
            predictions: Predictions::new(Value::new([0.0, 0.0]), 0.0),
            children: vec![
                EdgeMetrics::new(
                    "h2s".parse().unwrap(),
                    7,
                    MovesLeftPropagatedValue::new(0.0, 0.0),
                ),
                EdgeMetrics::new(
                    "ph2ss".parse().unwrap(),
                    2,
                    MovesLeftPropagatedValue::new(0.0, 0.0),
                ),
                EdgeMetrics::new(
                    "p".parse().unwrap(),
                    1,
                    MovesLeftPropagatedValue::new(0.0, 0.0),
                ),
            ],
        };

        let output = Mapper::new().metrics_to_policy_output(&game_state, &policy_metrics);
        let pass_idx = NUM_PIECE_MOVES + NUM_PUSH_PULL_MOVES;
        assert_eq!(output.len(), OUTPUT_SIZE);
        assert_eq!(
            output[0], 0.7,
            "Policy for h2s should be 7 visit / 10 visits"
        );
        assert_eq!(
            output[NUM_PIECE_MOVES], 0.2,
            "Policy for ph2ss should be 7 visit / 10 visits"
        );
        assert_eq!(
            output[pass_idx], 0.1,
            "Policy for pass should be 1 visit / 10 visits"
        );
    }

    #[test]
    fn test_policy_to_valid_actions() {
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
            .parse()
            .unwrap();

        let mut policy_scores = vec![f16::ZERO; OUTPUT_SIZE];

        policy_scores[map_action_to_policy_output_idx(&"a1n".parse().unwrap())] =
            f16::from_f32(1.0);
        let output = Mapper::new().policy_to_valid_actions(&game_state, &policy_scores);

        assert_eq!(output[0].action(), &"a1n".parse().unwrap());
        assert_eq!(output[0].policy_score(), f16::from_f32(0.04316948));
        assert_eq!(output[1].policy_score(), f16::from_f32(0.018761378));
        assert_eq!(output.len(), 52);
    }

    #[test]
    fn test_policy_to_valid_actions_2() {
        let mut game_state: GameState = "
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
            .parse()
            .unwrap();

        game_state = game_state.take_action(&"a1n".parse().unwrap());

        let mut policy_scores = vec![f16::ZERO; OUTPUT_SIZE];

        policy_scores[map_action_to_policy_output_idx(&"d4n".parse().unwrap())] =
            f16::from_f32(1.0);
        policy_scores[map_action_to_policy_output_idx(&"p".parse().unwrap())] = f16::from_f32(5.0);
        let output = Mapper::new().policy_to_valid_actions(&game_state, &policy_scores);

        assert_eq!(output[0].action(), &"d4n".parse().unwrap());
        assert_eq!(output[0].policy_score(), f16::from_f32(0.025623035));
        assert_eq!(output[1].policy_score(), f16::from_f32(0.011135726));
        assert_eq!(output[24].action(), &"p".parse().unwrap());
        assert_eq!(output[24].policy_score(), f16::from_f32(0.7182553));
        assert_eq!(output.len(), 25);
    }

    #[test]
    fn test_policy_to_valid_actions_as_silver() {
        let mut game_state: GameState = "
            1s
             +-----------------+
            8| r r             |
            7|                 |
            6|     x     x     |
            5|                 |
            4|       E         |
            3|     x     x     |
            2|                 |
            1| R               |
             +-----------------+
               a b c d e f g h"
            .parse()
            .unwrap();

        game_state = game_state.take_action(&"a8s".parse().unwrap());

        let mut policy_scores = vec![f16::ZERO; OUTPUT_SIZE];

        policy_scores
            [map_action_to_policy_output_idx(&"b8s".parse::<Action>().unwrap().rotate())] =
            f16::from_f32(1.0);
        let output = Mapper::new().policy_to_valid_actions(&game_state, &policy_scores);

        assert_eq!(output[0].action(), &"b8e".parse().unwrap());
        assert_eq!(output[0].policy_score(), f16::from_f32(0.07518246));
        assert_eq!(output[1].action(), &"b8s".parse().unwrap());
        assert_eq!(output[1].policy_score(), f16::from_f32(0.17299303));
        assert_eq!(output[11].policy_score(), f16::from_f32(0.07518246));
        assert_eq!(output.len(), 12);
    }

    #[test]
    fn test_policy_to_valid_actions_setup() {
        let game_state = GameState::initial();

        let game_state = take_actions!(game_state => a2);

        let mut policy_scores = vec![f16::ZERO; OUTPUT_SIZE];

        policy_scores[map_action_to_policy_output_idx(&"c2".parse::<Action>().unwrap())] =
            f16::from_f32(1.0);
        let output = Mapper::new().policy_to_valid_actions(&game_state, &policy_scores);

        assert_eq!(output[0].action(), &"b2".parse().unwrap());
        assert_eq!(output[0].policy_score(), f16::from_f32(0.06134603));
        assert_eq!(output[1].action(), &"c2".parse().unwrap());
        assert_eq!(output[1].policy_score(), f16::from_f32(0.14115575));
        assert_eq!(output[14].action(), &"h1".parse().unwrap());
        assert_eq!(output[14].policy_score(), f16::from_f32(0.06134603));
        assert_eq!(output.len(), 15);
    }

    #[test]
    fn test_policy_to_valid_actions_setup_silver() {
        let game_state = GameState::initial();

        let game_state = take_actions!(game_state => a2, b1, c1, d2, e2, f1, g1, h2);

        let mut policy_scores = vec![f16::ZERO; OUTPUT_SIZE];

        policy_scores[map_action_to_policy_output_idx(&"b8".parse::<Action>().unwrap().rotate())] =
            f16::from_f32(1.0);
        let output = Mapper::new().policy_to_valid_actions(&game_state, &policy_scores);

        assert_eq!(output[0].action(), &"a8".parse().unwrap());
        assert_eq!(output[0].policy_score(), f16::from_f32(0.057800222));
        assert_eq!(output[1].action(), &"b8".parse().unwrap());
        assert_eq!(output[1].policy_score(), f16::from_f32(0.13299692));
        assert_eq!(output[15].action(), &"h7".parse().unwrap());
        assert_eq!(output[15].policy_score(), f16::from_f32(0.057800222));
        assert_eq!(output.len(), 16);
    }

    // #[bench]
    // fn bench_game_state_to_input(b: &mut Bencher) {
    //     let game_state: GameState = "
    //             1s
    //              +-----------------+
    //             8| h c d m r d c h |
    //             7| r r r r e r r r |
    //             6|     x     x     |
    //             5|                 |
    //             4|                 |
    //             3|     x     x     |
    //             2| R R R R E R R R |
    //             1| H C D M R D C H |
    //              +-----------------+
    //                a b c d e f g h
    //         "
    //     .parse()
    //     .unwrap();

    //     b.iter(|| game_state_to_input_fn(&game_state, Mode::Train));
    // }

    // #[bench]
    // fn bench_game_state_to_input_multiple_actions(b: &mut Bencher) {
    //     let game_state: GameState = "
    //             1s
    //              +-----------------+
    //             8| h c d m r d c h |
    //             7| r r r r e r r r |
    //             6|     x     x     |
    //             5|                 |
    //             4|                 |
    //             3|     x     x     |
    //             2| R R R R E R R R |
    //             1| H C D M R D C H |
    //              +-----------------+
    //                a b c d e f g h
    //         "
    //     .parse()
    //     .unwrap();

    //     b.iter(|| {
    //         let game_state = game_state.take_action(&"d2n".parse().unwrap());
    //         game_state_to_input_fn(&game_state, Mode::Train);

    //         let game_state = game_state.take_action(&"e2n".parse().unwrap());
    //         game_state_to_input_fn(&game_state, Mode::Train);

    //         let game_state = game_state.take_action(&"b2n".parse().unwrap());
    //         game_state_to_input_fn(&game_state, Mode::Train);

    //         let game_state = game_state.take_action(&"f2n".parse().unwrap());
    //         game_state_to_input_fn(&game_state, Mode::Train);

    //         let game_state = game_state.take_action(&"d7s".parse().unwrap());
    //         game_state_to_input_fn(&game_state, Mode::Train);

    //         let game_state = game_state.take_action(&"e7s".parse().unwrap());
    //         game_state_to_input_fn(&game_state, Mode::Train);

    //         let game_state = game_state.take_action(&"f7s".parse().unwrap());
    //         game_state_to_input_fn(&game_state, Mode::Train);

    //         let game_state = game_state.take_action(&"g7s".parse().unwrap());
    //         game_state_to_input_fn(&game_state, Mode::Train)
    //     });
    // }
}
