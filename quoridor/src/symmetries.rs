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

    let symmetrical_state = game_state.vertical_symmetry();

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
    use engine::GameState as GameStateTrait;
    use itertools::Itertools;

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
    fn test_game_state_symmetry_initial() {
        let game_state: GameState = GameState::initial();

        let symmetries: Vec<GameState> = get_symmetries_game_state(game_state);
        let game_state_original = symmetries.first().unwrap();
        let game_state_symmetry = symmetries.last().unwrap();

        assert_eq!(
            game_state_original.to_string(),
            game_state_symmetry.to_string()
        );
    }

    #[test]
    fn test_game_state_symmetry_pawn_move() {
        let mut game_state: GameState = GameState::initial();
        game_state.take_action(&"d1".parse().unwrap());

        let game_state_rotated: &str = "
  +---+---+---+---+---+---+---+---+---+
9 |   |   |   |   | 2 |   |   |   |   |
  +---+---+---+---+---+---+---+---+---+
8 |   |   |   |   |   |   |   |   |   |
  +---+---+---+---+---+---+---+---+---+
7 |   |   |   |   |   |   |   |   |   |
  +---+---+---+---+---+---+---+---+---+
6 |   |   |   |   |   |   |   |   |   |
  +---+---+---+---+---+---+---+---+---+
5 |   |   |   |   |   |   |   |   |   |
  +---+---+---+---+---+---+---+---+---+
4 |   |   |   |   |   |   |   |   |   |
  +---+---+---+---+---+---+---+---+---+
3 |   |   |   |   |   |   |   |   |   |
  +---+---+---+---+---+---+---+---+---+
2 |   |   |   |   |   |   |   |   |   |
  +---+---+---+---+---+---+---+---+---+
1 |   |   |   |   |   | 1 |   |   |   |
  +---+---+---+---+---+---+---+---+---+
    a   b   c   d   e   f   g   h   i  

  P1: 10  P2: 10
";

        let symmetries: Vec<GameState> = get_symmetries_game_state(game_state);
        let game_state_original = symmetries.first().unwrap();
        let game_state_symmetry = symmetries.last().unwrap();

        assert_ne!(
            game_state_original.to_string(),
            game_state_symmetry.to_string()
        );

        assert_eq!(game_state_symmetry.to_string(), game_state_rotated)
    }

    #[test]
    fn test_game_state_symmetry_vertical_wall() {
        let mut game_state: GameState = GameState::initial();
        game_state.take_action(&"d2v".parse().unwrap());

        let game_state_rotated: &str = "
  +---+---+---+---+---+---+---+---+---+
9 |   |   |   |   | 2 |   |   |   |   |
  +---+---+---+---+---+---+---+---+---+
8 |   |   |   |   |   |   |   |   |   |
  +---+---+---+---+---+---+---+---+---+
7 |   |   |   |   |   |   |   |   |   |
  +---+---+---+---+---+---+---+---+---+
6 |   |   |   |   |   |   |   |   |   |
  +---+---+---+---+---+---+---+---+---+
5 |   |   |   |   |   |   |   |   |   |
  +---+---+---+---+---+---+---+---+---+
4 |   |   |   |   |   |   |   |   |   |
  +---+---+---+---+---+---+---+---+---+
3 |   |   |   |   |   |   |   |   |   |
  +---+---+---+---+---+---+---+---+---+
2 |   |   |   |   |   █   |   |   |   |
  +---+---+---+---+---█---+---+---+---+
1 |   |   |   |   | 1 █   |   |   |   |
  +---+---+---+---+---+---+---+---+---+
    a   b   c   d   e   f   g   h   i  

  P1: 9  P2: 10
";

        let symmetries: Vec<GameState> = get_symmetries_game_state(game_state);
        let game_state_original = symmetries.first().unwrap();
        let game_state_symmetry = symmetries.last().unwrap();

        assert_ne!(
            game_state_original.to_string(),
            game_state_symmetry.to_string()
        );

        assert_eq!(game_state_symmetry.to_string(), game_state_rotated)
    }

    #[test]
    fn test_game_state_symmetry_horizontal_wall() {
        let mut game_state: GameState = GameState::initial();
        game_state.take_action(&"d2h".parse().unwrap());

        let game_state_rotated: &str = "
  +---+---+---+---+---+---+---+---+---+
9 |   |   |   |   | 2 |   |   |   |   |
  +---+---+---+---+---+---+---+---+---+
8 |   |   |   |   |   |   |   |   |   |
  +---+---+---+---+---+---+---+---+---+
7 |   |   |   |   |   |   |   |   |   |
  +---+---+---+---+---+---+---+---+---+
6 |   |   |   |   |   |   |   |   |   |
  +---+---+---+---+---+---+---+---+---+
5 |   |   |   |   |   |   |   |   |   |
  +---+---+---+---+---+---+---+---+---+
4 |   |   |   |   |   |   |   |   |   |
  +---+---+---+---+---+---+---+---+---+
3 |   |   |   |   |   |   |   |   |   |
  +---+---+---+---+---+---+---+---+---+
2 |   |   |   |   |   |   |   |   |   |
  +---+---+---+---+■■■■■■■+---+---+---+
1 |   |   |   |   | 1 |   |   |   |   |
  +---+---+---+---+---+---+---+---+---+
    a   b   c   d   e   f   g   h   i  

  P1: 9  P2: 10
";

        let symmetries: Vec<GameState> = get_symmetries_game_state(game_state);
        let game_state_original = symmetries.first().unwrap();
        let game_state_symmetry = symmetries.last().unwrap();

        assert_ne!(
            game_state_original.to_string(),
            game_state_symmetry.to_string()
        );

        assert_eq!(game_state_symmetry.to_string(), game_state_rotated)
    }

    #[test]
    fn test_game_state_symmetry_horizontal_wall_2() {
        let mut game_state: GameState = GameState::initial();
        game_state.take_action(&"a4h".parse().unwrap());

        let game_state_rotated: &str = "
  +---+---+---+---+---+---+---+---+---+
9 |   |   |   |   | 2 |   |   |   |   |
  +---+---+---+---+---+---+---+---+---+
8 |   |   |   |   |   |   |   |   |   |
  +---+---+---+---+---+---+---+---+---+
7 |   |   |   |   |   |   |   |   |   |
  +---+---+---+---+---+---+---+---+---+
6 |   |   |   |   |   |   |   |   |   |
  +---+---+---+---+---+---+---+---+---+
5 |   |   |   |   |   |   |   |   |   |
  +---+---+---+---+---+---+---+---+---+
4 |   |   |   |   |   |   |   |   |   |
  +---+---+---+---+---+---+---+■■■■■■■+
3 |   |   |   |   |   |   |   |   |   |
  +---+---+---+---+---+---+---+---+---+
2 |   |   |   |   |   |   |   |   |   |
  +---+---+---+---+---+---+---+---+---+
1 |   |   |   |   | 1 |   |   |   |   |
  +---+---+---+---+---+---+---+---+---+
    a   b   c   d   e   f   g   h   i  

  P1: 9  P2: 10
";

        let symmetries: Vec<GameState> = get_symmetries_game_state(game_state);
        let game_state_original = symmetries.first().unwrap();
        let game_state_symmetry = symmetries.last().unwrap();

        assert_ne!(
            game_state_original.to_string(),
            game_state_symmetry.to_string()
        );

        assert_eq!(game_state_symmetry.to_string(), game_state_rotated)
    }

    #[test]
    fn test_game_state_symmetry_actions() {
        let game_state: GameState = GameState::initial();

        let symmetries = get_symmetries(PositionMetrics {
            game_state,
            policy: NodeMetrics {
                visits: 0,
                value: Value::new([0.0, 0.0]),
                moves_left: 0.0,
                children: vec![
                    NodeChildMetrics::new("a9".parse().unwrap(), 0.0, 0.0, 0),
                    NodeChildMetrics::new("b9".parse().unwrap(), 0.0, 0.0, 0),
                    NodeChildMetrics::new("h9".parse().unwrap(), 0.0, 0.0, 0),
                    NodeChildMetrics::new("a1".parse().unwrap(), 0.0, 0.0, 0),
                    NodeChildMetrics::new("a1v".parse().unwrap(), 0.0, 0.0, 0),
                    NodeChildMetrics::new("a1h".parse().unwrap(), 0.0, 0.0, 0),
                    NodeChildMetrics::new("h3h".parse().unwrap(), 0.0, 0.0, 0),
                    NodeChildMetrics::new("h3v".parse().unwrap(), 0.0, 0.0, 0),
                ],
            },
            score: Value::new([0.0, 0.0]),
            moves_left: 0,
        });

        let node_metrics = &symmetries.last().unwrap().policy.children;
        assert_eq!(node_metrics[0].action(), &"i9".parse::<Action>().unwrap());
        assert_eq!(node_metrics[1].action(), &"h9".parse::<Action>().unwrap());
        assert_eq!(node_metrics[2].action(), &"b9".parse::<Action>().unwrap());
        assert_eq!(node_metrics[3].action(), &"i1".parse::<Action>().unwrap());
        assert_eq!(node_metrics[4].action(), &"h1v".parse::<Action>().unwrap());
        assert_eq!(node_metrics[5].action(), &"h1h".parse::<Action>().unwrap());
        assert_eq!(node_metrics[6].action(), &"a3h".parse::<Action>().unwrap());
        assert_eq!(node_metrics[7].action(), &"a3v".parse::<Action>().unwrap());
    }

    #[test]
    fn test_action_node_symmetry_all_actions() {
        let game_state: GameState = GameState::initial();
        let move_actions = || {
            (1..=9)
                .flat_map(|row| ('a'..='i').map(move |col| (row, col)))
                .map(|(row, col)| format!("{col}{row}"))
                .map(|str| str.parse::<Action>().unwrap())
        };

        let wall_actions = || {
            (1..9)
                .flat_map(|row| ('a'..'i').map(move |col| (row, col)))
                .flat_map(|(row, col)| {
                    ["v", "h"]
                        .into_iter()
                        .map(move |suffix| format!("{col}{row}{suffix}"))
                })
                .map(|str| str.parse::<Action>().unwrap())
        };

        let actions = || move_actions().chain(wall_actions());

        let children = actions()
            .map(|action| NodeChildMetrics::new(action, 0.0, 0.0, 0))
            .collect_vec();

        let symmetries = get_symmetries(PositionMetrics {
            game_state,
            policy: NodeMetrics {
                visits: 0,
                value: Value::new([0.0, 0.0]),
                moves_left: 0.0,
                children,
            },
            score: Value::new([0.0, 0.0]),
            moves_left: 0,
        });

        for (action, symmetrical_action) in actions().zip(
            symmetries
                .last()
                .unwrap()
                .policy
                .children
                .iter()
                .map(|m| m.action()),
        ) {
            assert_eq!(action, symmetrical_action.vertical_symmetry());
            assert_eq!(action.vertical_symmetry(), *symmetrical_action);
        }
    }
}
