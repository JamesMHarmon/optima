use model::node_metrics::NodeMetrics;
use model::position_metrics::PositionMetrics;
use common::linked_list::List;

use super::value::Value;
use super::zobrist::Zobrist;
use super::engine::{GameState,Phase,PlayPhase,PushPullState,PieceBoard,PieceBoardState};
use super::action::{Action,Piece,map_bit_board_to_squares};

pub fn get_symmetries(metrics: PositionMetrics<GameState,Action,Value>) -> Vec<PositionMetrics<GameState,Action,Value>> {
    let PositionMetrics { game_state, policy, score } = metrics;

    get_symmetries_game_state(game_state).into_iter()
        .zip(get_symmetries_node_metrics(policy))
        .map(|(game_state, policy)| PositionMetrics {
            game_state,
            policy,
            score: score.clone(),
        })
        .collect()
}

fn get_symmetries_node_metrics(metrics: NodeMetrics<Action>) -> Vec<NodeMetrics<Action>> {
    let children_visits_symmetry = metrics.children_visits.iter()
        .map(|(a, visits)| (a.invert_horizontal(), *visits))
        .collect();

    let metrics_symmetry = NodeMetrics {
        W: metrics.W,
        visits: metrics.visits,
        children_visits: children_visits_symmetry
    };

    vec!(metrics, metrics_symmetry)
}

fn get_symmetries_game_state(game_state: GameState) -> Vec<GameState> {
    match game_state.as_play_phase() {
        Some(_) => {
            let transposition = get_horizontal_symmetry(&game_state);
            vec!(game_state, transposition)
        },
        None => vec!(game_state)
    }
}

// The hash history produced for Zobrist are not accurate, this assumes that the hashes do not matter.
fn get_horizontal_symmetry(game_state: &GameState) -> GameState {
    if let Some(play_phase) = game_state.as_play_phase() {
        let previous_piece_boards_this_move = play_phase.get_previous_piece_boards().iter()
            .map(|b| b.get_piece_board())
            .map(get_piece_board_horizontal_symmetry)
            .collect::<Vec<_>>();

        let p1_turn_to_move = game_state.is_p1_turn_to_move();
        let step_num = game_state.get_current_step();
        let piece_board = get_piece_board_horizontal_symmetry(game_state.get_piece_board());
        let hash = Zobrist::from_piece_board(piece_board.get_piece_board(), p1_turn_to_move, step_num);
        let push_pull_state = play_phase.get_push_pull_state();
        let initial_hash_of_move = previous_piece_boards_this_move.first()
            .map(|b| Zobrist::from_piece_board(b.get_piece_board(), p1_turn_to_move, 0))
            .unwrap_or(hash);

        let play_phase = PlayPhase::new(
            initial_hash_of_move,
            List::new(),
            previous_piece_boards_this_move,
            get_push_pull_state_horizontal_symmetry(push_pull_state),
            play_phase.get_piece_trapped_this_turn()
        );

        GameState::new(
            p1_turn_to_move, game_state.get_move_number(),
            Phase::PlayPhase(play_phase),
            piece_board,
            hash
        )
    } else {
        panic!("Cannot create vertical symmetry for place phase");
    }
}

fn get_push_pull_state_horizontal_symmetry(push_pull_state: PushPullState) -> PushPullState {
    match push_pull_state {
        PushPullState::MustCompletePush(square, piece) => PushPullState::MustCompletePush(square.invert_horizontal(), piece),
        PushPullState::PossiblePull(square, piece) => PushPullState::PossiblePull(square.invert_horizontal(), piece),
        PushPullState::None => PushPullState::None,
    }
}

fn get_piece_board_horizontal_symmetry(piece_board: &PieceBoardState) -> PieceBoard {
    PieceBoard::new(
        get_bit_board_horizontal_symmetry(piece_board.get_player_piece_mask(true)),
        get_bit_board_horizontal_symmetry(piece_board.get_bits_by_piece_type(Piece::Elephant)),
        get_bit_board_horizontal_symmetry(piece_board.get_bits_by_piece_type(Piece::Camel)),
        get_bit_board_horizontal_symmetry(piece_board.get_bits_by_piece_type(Piece::Horse)),
        get_bit_board_horizontal_symmetry(piece_board.get_bits_by_piece_type(Piece::Dog)),
        get_bit_board_horizontal_symmetry(piece_board.get_bits_by_piece_type(Piece::Cat)),
        get_bit_board_horizontal_symmetry(piece_board.get_bits_by_piece_type(Piece::Rabbit)),
    )
}

fn get_bit_board_horizontal_symmetry(bit_board: u64) -> u64 {
    map_bit_board_to_squares(bit_board).iter()
        .map(|s| s.invert_horizontal())
        .fold(0, |r, s| r | s.as_bit_board())
}

#[cfg(test)]
mod tests {
    extern crate test;

    use super::*;
    use super::super::action::{Action,Direction,Piece,Square};

    fn step_num_symmetry_to_string(game_state: GameState, step_num: usize) -> String {
        let symmetries = get_symmetries_game_state(game_state);
        let game_state = symmetries.first().unwrap();
        let game_state_symmetry = symmetries.last().unwrap();
        let piece_board = game_state_symmetry.get_piece_board_for_step(step_num);
        let game_state_symmetry_step = GameState::new(
            game_state.is_p1_turn_to_move(),
            game_state.get_move_number(),
            Phase::PlayPhase(PlayPhase::initial(Zobrist::initial(), List::new())),
            PieceBoard::new(
                piece_board.get_player_piece_mask(true), 
                piece_board.get_bits_by_piece_type(Piece::Elephant), 
                piece_board.get_bits_by_piece_type(Piece::Camel), 
                piece_board.get_bits_by_piece_type(Piece::Horse), 
                piece_board.get_bits_by_piece_type(Piece::Dog), 
                piece_board.get_bits_by_piece_type(Piece::Cat), 
                piece_board.get_bits_by_piece_type(Piece::Rabbit)
            ),
            Zobrist::initial()
        );

        game_state_symmetry_step.to_string()
    }

    #[test]
    fn test_game_state_symmetry_step_0() {
        let game_state: GameState = "
            5g
             +-----------------+
            8|   r     r   r   |
            7|                 |
            6|     x     x     |
            5|     E r         |
            4|                 |
            3|     x     x     |
            2|                 |
            1| R         M     |
             +-----------------+
               a b c d e f g h"
            .parse().unwrap();

        let symmetries = get_symmetries_game_state(game_state);
        let game_state_original = symmetries.first().unwrap();
        let game_state_symmetry = symmetries.last().unwrap();

        assert_eq!(game_state_original.to_string(), "5g
 +-----------------+
8|   r     r   r   |
7|                 |
6|     x     x     |
5|     E r         |
4|                 |
3|     x     x     |
2|                 |
1| R         M     |
 +-----------------+
   a b c d e f g h
");

assert_eq!(game_state_symmetry.to_string(), "5g
 +-----------------+
8|   r   r     r   |
7|                 |
6|     x     x     |
5|         r E     |
4|                 |
3|     x     x     |
2|                 |
1|     M         R |
 +-----------------+
   a b c d e f g h
");
    }

    #[test]
    fn test_game_state_symmetry_step_1() {
        let game_state: GameState = "
            5g
             +-----------------+
            8|   r     r   r   |
            7|                 |
            6|     x     x     |
            5|     E r         |
            4|                 |
            3|     x     x     |
            2|                 |
            1| R         M     |
             +-----------------+
               a b c d e f g h"
            .parse().unwrap();

        let game_state = game_state.take_action(&Action::Move(Square::new('a', 1), Direction::Up));

        let game_state_symmetry_step_0 = step_num_symmetry_to_string(game_state.clone(), 0);
        let game_state_symmetry_step_1 = step_num_symmetry_to_string(game_state, 1);

assert_eq!(game_state_symmetry_step_0, "5g
 +-----------------+
8|   r   r     r   |
7|                 |
6|     x     x     |
5|         r E     |
4|                 |
3|     x     x     |
2|                 |
1|     M         R |
 +-----------------+
   a b c d e f g h
");

assert_eq!(game_state_symmetry_step_1, "5g
 +-----------------+
8|   r   r     r   |
7|                 |
6|     x     x     |
5|         r E     |
4|                 |
3|     x     x     |
2|               R |
1|     M           |
 +-----------------+
   a b c d e f g h
");
    }

    #[test]
    fn test_game_state_symmetry_step_4() {
        let game_state: GameState = "
            5g
             +-----------------+
            8|   r     r   r   |
            7|                 |
            6|     x     x     |
            5|     E r         |
            4|                 |
            3|     x     x     |
            2|                 |
            1| R         M     |
             +-----------------+
               a b c d e f g h"
            .parse().unwrap();

        let game_state = game_state.take_action(&Action::Move(Square::new('a', 1), Direction::Up));
        let game_state = game_state.take_action(&Action::Move(Square::new('a', 2), Direction::Right));
        let game_state = game_state.take_action(&Action::Move(Square::new('b', 2), Direction::Right));

        let game_state_symmetry_step_0 = step_num_symmetry_to_string(game_state.clone(), 0);
        let game_state_symmetry_step_1 = step_num_symmetry_to_string(game_state.clone(), 1);
        let game_state_symmetry_step_2 = step_num_symmetry_to_string(game_state.clone(), 2);
        let game_state_symmetry_step_3 = step_num_symmetry_to_string(game_state, 3);

assert_eq!(game_state_symmetry_step_0, "5g
 +-----------------+
8|   r   r     r   |
7|                 |
6|     x     x     |
5|         r E     |
4|                 |
3|     x     x     |
2|                 |
1|     M         R |
 +-----------------+
   a b c d e f g h
");

assert_eq!(game_state_symmetry_step_1, "5g
 +-----------------+
8|   r   r     r   |
7|                 |
6|     x     x     |
5|         r E     |
4|                 |
3|     x     x     |
2|               R |
1|     M           |
 +-----------------+
   a b c d e f g h
");

assert_eq!(game_state_symmetry_step_2, "5g
 +-----------------+
8|   r   r     r   |
7|                 |
6|     x     x     |
5|         r E     |
4|                 |
3|     x     x     |
2|             R   |
1|     M           |
 +-----------------+
   a b c d e f g h
");

assert_eq!(game_state_symmetry_step_3, "5g
 +-----------------+
8|   r   r     r   |
7|                 |
6|     x     x     |
5|         r E     |
4|                 |
3|     x     x     |
2|           R     |
1|     M           |
 +-----------------+
   a b c d e f g h
");
    }
}