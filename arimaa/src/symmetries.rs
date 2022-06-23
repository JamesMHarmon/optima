use model::{node_metrics::NodeMetrics, NodeChildMetrics};
use model::position_metrics::PositionMetrics;

use super::{Action, GameState, Value};

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

fn get_symmetries_node_metrics(metrics: NodeMetrics<Action, Value>) -> Vec<NodeMetrics<Action, Value>> {
    let children_symmetry = metrics
        .children
        .iter()
        .map(|m| NodeChildMetrics::new(m.action().invert_horizontal(), m.Q(), m.M(), m.visits()))
        .collect();

    let metrics_symmetry = NodeMetrics {
        visits: metrics.visits,
        value: metrics.value.clone(),
        children: children_symmetry,
    };

    vec![metrics, metrics_symmetry]
}

fn get_symmetries_game_state(game_state: GameState) -> Vec<GameState> {
    if game_state.is_play_phase() {
        let symmetrical_state = game_state.get_vertical_symmetry();
        return vec![game_state, symmetrical_state];
    }

    vec![game_state]
}

#[cfg(test)]
mod tests {
    extern crate test;

    use super::*;
    use arimaa_engine::Action;
    use model::NodeChildMetrics;

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
            get_symmetries_game_state(game_state.clone())[1].to_string(),
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

        for (
            original,
            symmetrical,
        ) in original_children.into_iter().zip(symmetrical_children)
        {
            match original.action() {
                Action::Move(original_square, original_direction) => match symmetrical.action() {
                    Action::Move(symmetrical_square, symmetrical_direction) => {
                        assert_eq!(*original_square, symmetrical_square.invert_horizontal());
                        assert_eq!(
                            *original_direction,
                            symmetrical_direction.invert_horizontal()
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
}
