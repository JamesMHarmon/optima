use model::position_metrics::PositionMetrics;
use model::{node_metrics::NodeMetrics, NodeChildMetrics};

use super::{Action, GameState, Value};

pub fn get_symmetries(
    metrics: PositionMetrics<GameState, Action, Value>,
) -> Vec<PositionMetrics<GameState, Action, Value>> {
    let PositionMetrics {
        game_state,
        policy,
        score,
        moves_left,
    } = &metrics;

    let symmetrical_state = game_state.get_vertical_symmetry();

    let symmetrical_metrics = PositionMetrics {
        game_state: symmetrical_state,
        policy: symmetrical_node_metrics(policy),
        score: score.clone(),
        moves_left: *moves_left,
    };

    vec![metrics, symmetrical_metrics]
}

fn symmetrical_node_metrics(metrics: &NodeMetrics<Action, Value>) -> NodeMetrics<Action, Value> {
    let children_symmetry = metrics
        .children
        .iter()
        .map(|m| NodeChildMetrics::new(m.action().vertical_symmetry(), m.Q(), m.M(), m.visits()))
        .collect();

    NodeMetrics {
        visits: metrics.visits,
        value: metrics.value.clone(),
        moves_left: metrics.moves_left,
        children: children_symmetry,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::Value;
    use arimaa_engine::{take_actions, Action};
    use engine::GameState as GameStateTrait;
    use model::NodeChildMetrics;

    fn get_symmetries_game_state(game_state: GameState) -> Vec<GameState> {
        let symmetries = get_symmetries(PositionMetrics {
            game_state,
            policy: NodeMetrics {
                visits: 0,
                value: Value::new([0.0, 0.0]),
                moves_left: 0.0,
                children: vec![],
            },
            score: Value::new([0.0, 0.0]),
            moves_left: 0,
        });

        symmetries.into_iter().map(|s| s.game_state).collect()
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

        let game_state = game_state.take_action(&"a1n".parse().unwrap());

        assert_eq!(
            get_symmetries_game_state(game_state)[1].to_string(),
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

        assert_eq!(
            get_symmetries_game_state(game_state.clone())[1].to_string(),
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

        let game_state = game_state.take_action(&"a1n".parse().unwrap());

        assert_eq!(
            get_symmetries_game_state(game_state.clone())[1].to_string(),
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
        let game_state = game_state.take_action(&"a2e".parse().unwrap());

        assert_eq!(
            get_symmetries_game_state(game_state.clone())[1].to_string(),
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
        let game_state = game_state.take_action(&"b2e".parse().unwrap());

        assert_eq!(
            get_symmetries_game_state(game_state)[1].to_string(),
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

        let game_state = game_state.take_action(&"a1n".parse().unwrap());
        let game_state = game_state.take_action(&"a2e".parse().unwrap());
        let game_state = game_state.take_action(&"b2e".parse().unwrap());

        let mut symmetries = get_symmetries(PositionMetrics {
            score: [0.0, 1.0].into(),
            game_state,
            policy: NodeMetrics {
                visits: 800,
                value: Value::new([0.0, 0.0]),
                moves_left: 0.0,
                children: vec![
                    NodeChildMetrics::new("c2n".parse().unwrap(), 0.0, 0.0, 500),
                    NodeChildMetrics::new("a2e".parse().unwrap(), 0.0, 0.0, 250),
                    NodeChildMetrics::new("c2w".parse().unwrap(), 0.0, 0.0, 50),
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
                    ..
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
                    ..
                },
            ..
        } = symmetries.pop().unwrap();

        assert_eq!(symmetrical_score, original_score);
        assert_eq!(symmetrical_visits, original_visits);
        assert_eq!(original_children.len(), symmetrical_children.len());

        for (original, symmetrical) in original_children.into_iter().zip(symmetrical_children) {
            match original.action() {
                Action::Move(original_square, original_direction) => match symmetrical.action() {
                    Action::Move(symmetrical_square, symmetrical_direction) => {
                        assert_eq!(*original_square, symmetrical_square.vertical_symmetry());
                        assert_eq!(
                            *original_direction,
                            symmetrical_direction.vertical_symmetry()
                        );
                    }
                    _ => panic!(),
                },
                _ => panic!(),
            }

            assert_eq!(original.visits(), symmetrical.visits());
            assert_eq!(original.Q(), symmetrical.Q());
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

    #[test]
    #[allow(non_snake_case)]
    fn test_symmetries_setup() {
        let game_state = GameState::initial();

        let game_state = take_actions!(game_state => a1, b2, e1);

        let mut symmetries = get_symmetries(PositionMetrics {
            score: [0.0, 1.0].into(),
            game_state,
            policy: NodeMetrics {
                visits: 800,
                value: Value::new([0.0, 0.0]),
                moves_left: 0.0,
                children: vec![
                    NodeChildMetrics::new("a2".parse().unwrap(), 0.0, 0.0, 500),
                    NodeChildMetrics::new("c2".parse().unwrap(), 0.0, 0.0, 250),
                    NodeChildMetrics::new("d1".parse().unwrap(), 0.0, 0.0, 50),
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
                    ..
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
                    ..
                },
            ..
        } = symmetries.pop().unwrap();

        assert_eq!(symmetrical_score, original_score);
        assert_eq!(symmetrical_visits, original_visits);
        assert_eq!(original_children.len(), symmetrical_children.len());

        for (original, symmetrical) in original_children.into_iter().zip(symmetrical_children) {
            match original.action() {
                Action::Place(original_square) => match symmetrical.action() {
                    Action::Place(symmetrical_square) => {
                        assert_eq!(*original_square, symmetrical_square.vertical_symmetry());
                    }
                    _ => panic!(),
                },
                _ => panic!(),
            }

            assert_eq!(original.visits(), symmetrical.visits());
            assert_eq!(original.Q(), symmetrical.Q());
        }

        assert_eq!(
            original_game_state.to_string(),
            "1g
 +-----------------+
8|                 |
7|                 |
6|     x     x     |
5|                 |
4|                 |
3|     x     x     |
2|   M             |
1| E       H       |
 +-----------------+
   a b c d e f g h
"
        );
        assert_eq!(
            symmetrical_game_state.to_string(),
            "1g
 +-----------------+
8|                 |
7|                 |
6|     x     x     |
5|                 |
4|                 |
3|     x     x     |
2|             M   |
1|       H       E |
 +-----------------+
   a b c d e f g h
"
        );
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_symmetries_setup_silver() {
        let game_state = GameState::initial();

        let game_state = take_actions!(game_state => a1, b2, e1, c2, d2, e2, a2, f1, h7, d8, g7);

        let mut symmetries = get_symmetries(PositionMetrics {
            score: [0.0, 1.0].into(),
            game_state,
            policy: NodeMetrics {
                visits: 800,
                value: Value::new([0.0, 0.0]),
                moves_left: 0.0,
                children: vec![
                    NodeChildMetrics::new("g7".parse().unwrap(), 0.0, 0.0, 500),
                    NodeChildMetrics::new("d8".parse().unwrap(), 0.0, 0.0, 250),
                    NodeChildMetrics::new("e7".parse().unwrap(), 0.0, 0.0, 50),
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
                    ..
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
                    ..
                },
            ..
        } = symmetries.pop().unwrap();

        assert_eq!(symmetrical_score, original_score);
        assert_eq!(symmetrical_visits, original_visits);
        assert_eq!(original_children.len(), symmetrical_children.len());

        for (original, symmetrical) in original_children.into_iter().zip(symmetrical_children) {
            match original.action() {
                Action::Place(original_square) => match symmetrical.action() {
                    Action::Place(symmetrical_square) => {
                        assert_eq!(*original_square, symmetrical_square.vertical_symmetry());
                    }
                    _ => panic!(),
                },
                _ => panic!(),
            }

            assert_eq!(original.visits(), symmetrical.visits());
            assert_eq!(original.Q(), symmetrical.Q());
        }

        assert_eq!(
            original_game_state.to_string(),
            "1s
 +-----------------+
8|       m         |
7|             h e |
6|     x     x     |
5|                 |
4|                 |
3|     x     x     |
2| C M H D D R R R |
1| E R R R H C R R |
 +-----------------+
   a b c d e f g h
"
        );
        assert_eq!(
            symmetrical_game_state.to_string(),
            "1s
 +-----------------+
8|         m       |
7| e h             |
6|     x     x     |
5|                 |
4|                 |
3|     x     x     |
2| R R R D D H M C |
1| R R C H R R R E |
 +-----------------+
   a b c d e f g h
"
        );
    }
}
