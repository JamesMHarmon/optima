use common::linked_list::List;
use model::node_metrics::NodeMetrics;
use model::position_metrics::PositionMetrics;

use arimaa_engine::action::{map_bit_board_to_squares, Action, Piece};
use arimaa_engine::{GameState, Value};

pub fn get_symmetries(
    metrics: PositionMetrics<GameState, Action, Value>,
) -> Vec<PositionMetrics<GameState, Action, Value>> {
    let PositionMetrics {
        game_state,
        policy,
        score,
        moves_left,
    } = metrics;

    get_symmetries_game_state(game_state)
        .into_iter()
        .zip(get_symmetries_node_metrics(policy))
        .map(|(game_state, policy)| PositionMetrics {
            game_state,
            policy,
            score: score.clone(),
            moves_left,
        })
        .collect()
}

fn get_symmetries_node_metrics(metrics: NodeMetrics<Action>) -> Vec<NodeMetrics<Action>> {
    let children_symmetry = metrics
        .children
        .iter()
        .map(|(a, w, visits)| (a.invert_horizontal(), *w, *visits))
        .collect();

    let metrics_symmetry = NodeMetrics {
        visits: metrics.visits,
        children: children_symmetry,
    };

    vec![metrics, metrics_symmetry]
}

fn get_symmetries_game_state(game_state: GameState) -> Vec<GameState> {
    match game_state.as_play_phase() {
        Some(_) => {
            let transposition = get_horizontal_symmetry(&game_state);
            vec![game_state, transposition]
        }
        None => vec![game_state],
    }
}

// The hash history produced for Zobrist are not accurate, this assumes that the hashes do not matter.
fn get_horizontal_symmetry(game_state: &GameState) -> GameState {
    if let Some(play_phase) = game_state.as_play_phase() {
        let previous_piece_boards_this_move = play_phase
            .get_previous_piece_boards()
            .iter()
            .map(|b| b.get_piece_board())
            .map(get_piece_board_horizontal_symmetry)
            .collect::<Vec<_>>();

        let p1_turn_to_move = game_state.is_p1_turn_to_move();
        let step_num = game_state.get_current_step();
        let piece_board = get_piece_board_horizontal_symmetry(game_state.get_piece_board());
        let hash =
            Zobrist::from_piece_board(piece_board.get_piece_board(), p1_turn_to_move, step_num);
        let push_pull_state = play_phase.get_push_pull_state();
        let initial_hash_of_move = previous_piece_boards_this_move
            .first()
            .map(|b| Zobrist::from_piece_board(b.get_piece_board(), p1_turn_to_move, 0))
            .unwrap_or(hash);

        let play_phase = PlayPhase::new(
            initial_hash_of_move,
            List::new(),
            previous_piece_boards_this_move,
            get_push_pull_state_horizontal_symmetry(push_pull_state),
            play_phase.get_piece_trapped_this_turn(),
        );

        GameState::new(
            p1_turn_to_move,
            game_state.get_move_number(),
            Phase::PlayPhase(play_phase),
            piece_board,
            hash,
        )
    } else {
        panic!("Cannot create vertical symmetry for place phase");
    }
}

fn get_push_pull_state_horizontal_symmetry(push_pull_state: PushPullState) -> PushPullState {
    match push_pull_state {
        PushPullState::MustCompletePush(square, piece) => {
            PushPullState::MustCompletePush(square.invert_horizontal(), piece)
        }
        PushPullState::PossiblePull(square, piece) => {
            PushPullState::PossiblePull(square.invert_horizontal(), piece)
        }
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
    map_bit_board_to_squares(bit_board)
        .iter()
        .map(|s| s.invert_horizontal())
        .fold(0, |r, s| r | s.as_bit_board())
}

#[cfg(test)]
mod tests {
    extern crate test;

    use super::*;
    use arimaa_engine::{Action, Direction, Piece, Square};

    fn step_num_symmetry_to_string(game_state: GameState, step_num: usize) -> String {
        let symmetries = get_symmetries_game_state(game_state);
        let game_state = symmetries.first().unwrap();
        let game_state_symmetry = symmetries.last().unwrap();
        let piece_board = game_state_symmetry.get_piece_board_for_step(step_num);
        let game_state_symmetry_step = "";

        // GameState::new(
        //     game_state.is_p1_turn_to_move(),
        //     game_state.get_move_number(),
        //     Phase::PlayPhase(PlayPhase::initial(Zobrist::initial(), List::new())),
        //     PieceBoard::new(
        //         piece_board.get_player_piece_mask(true),
        //         piece_board.get_bits_by_piece_type(Piece::Elephant),
        //         piece_board.get_bits_by_piece_type(Piece::Camel),
        //         piece_board.get_bits_by_piece_type(Piece::Horse),
        //         piece_board.get_bits_by_piece_type(Piece::Dog),
        //         piece_board.get_bits_by_piece_type(Piece::Cat),
        //         piece_board.get_bits_by_piece_type(Piece::Rabbit),
        //     ),
        //     Zobrist::initial(),
        // );

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
            .parse()
            .unwrap();

        let symmetries = get_symmetries_game_state(game_state);
        let game_state_original = symmetries.first().unwrap();
        let game_state_symmetry = symmetries.last().unwrap();

        assert_eq!(
            game_state_original.to_string(),
            "5g
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
"
        );

        assert_eq!(
            game_state_symmetry.to_string(),
            "5g
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
"
        );
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
            .parse()
            .unwrap();

        let game_state = game_state.take_action(&Action::Move(Square::new('a', 1), Direction::Up));

        let game_state_symmetry_step_0 = step_num_symmetry_to_string(game_state.clone(), 0);
        let game_state_symmetry_step_1 = step_num_symmetry_to_string(game_state, 1);

        assert_eq!(
            game_state_symmetry_step_0,
            "5g
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
"
        );

        assert_eq!(
            game_state_symmetry_step_1,
            "5g
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
"
        );
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
            .parse()
            .unwrap();

        let game_state = game_state.take_action(&Action::Move(Square::new('a', 1), Direction::Up));
        let game_state =
            game_state.take_action(&Action::Move(Square::new('a', 2), Direction::Right));
        let game_state =
            game_state.take_action(&Action::Move(Square::new('b', 2), Direction::Right));

        let game_state_symmetry_step_0 = step_num_symmetry_to_string(game_state.clone(), 0);
        let game_state_symmetry_step_1 = step_num_symmetry_to_string(game_state.clone(), 1);
        let game_state_symmetry_step_2 = step_num_symmetry_to_string(game_state.clone(), 2);
        let game_state_symmetry_step_3 = step_num_symmetry_to_string(game_state, 3);

        assert_eq!(
            game_state_symmetry_step_0,
            "5g
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
"
        );

        assert_eq!(
            game_state_symmetry_step_1,
            "5g
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
"
        );

        assert_eq!(
            game_state_symmetry_step_2,
            "5g
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
"
        );

        assert_eq!(
            game_state_symmetry_step_3,
            "5g
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
"
        );
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_get_symmetries() {
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
            .parse()
            .unwrap();

        let game_state = game_state.take_action(&Action::Move(Square::new('a', 1), Direction::Up));
        let game_state =
            game_state.take_action(&Action::Move(Square::new('a', 2), Direction::Right));
        let game_state =
            game_state.take_action(&Action::Move(Square::new('b', 2), Direction::Right));

        let mut symmetries = get_symmetries(PositionMetrics {
            score: Value([0.0, 1.0]),
            game_state,
            policy: NodeMetrics {
                visits: 800,
                children: vec![
                    (Action::Move(Square::new('c', 2), Direction::Up), 0.0, 500),
                    (
                        Action::Move(Square::new('c', 2), Direction::Right),
                        0.0,
                        250,
                    ),
                    (Action::Move(Square::new('c', 2), Direction::Left), 0.0, 50),
                ],
            },
            moves_left: 0,
        });

        let PositionMetrics {
            score: symmetrical_score,
            game_state: symmetrical_game_state,
            policy:
                NodeMetrics {
                    visits: symmetrical_visits,
                    children: symmetrical_children,
                },
            ..
        } = symmetries.pop().unwrap();

        let PositionMetrics {
            score: original_score,
            game_state: original_game_state,
            policy:
                NodeMetrics {
                    visits: original_visits,
                    children: original_children,
                },
            ..
        } = symmetries.pop().unwrap();

        assert_eq!(symmetrical_score, original_score);
        assert_eq!(symmetrical_visits, original_visits);
        assert_eq!(original_children.len(), symmetrical_children.len());

        for (
            (original_action, original_w, original_visits),
            (symmetrical_action, symmetrical_w, symmetrical_visits),
        ) in original_children.into_iter().zip(symmetrical_children)
        {
            match original_action {
                Action::Move(original_square, original_direction) => match symmetrical_action {
                    Action::Move(symmetrical_square, symmetrical_direction) => {
                        assert_eq!(original_square, symmetrical_square.invert_horizontal());
                        assert_eq!(
                            original_direction,
                            symmetrical_direction.invert_horizontal()
                        );
                    }
                    _ => panic!(),
                },
                _ => panic!(),
            }

            assert_eq!(original_visits, symmetrical_visits);
            assert_eq!(original_w, symmetrical_w);
        }

        assert_eq!(
            original_game_state.to_string(),
            "5g
 +-----------------+
8|   r     r   r   |
7|                 |
6|     x     x     |
5|     E r         |
4|                 |
3|     x     x     |
2|     R           |
1|           M     |
 +-----------------+
   a b c d e f g h
"
        );

        assert_eq!(
            symmetrical_game_state.to_string(),
            "5g
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
"
        );
    }
}
