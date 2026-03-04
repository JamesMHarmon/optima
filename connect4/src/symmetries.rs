use super::{Action, GameState, Predictions};
use model::{EdgeMetrics, NodeMetrics, PositionMetrics};
use puct::MovesLeftSnapshot;

pub fn get_symmetries(
    metrics: PositionMetrics<GameState, Action, Predictions, MovesLeftSnapshot>,
) -> Vec<PositionMetrics<GameState, Action, Predictions, MovesLeftSnapshot>> {
    let PositionMetrics {
        game_state,
        node_metrics,
    } = &metrics;

    let symmetrical_state = game_state.horizontal_symmetry();

    let symmetrical_metrics = PositionMetrics {
        game_state: symmetrical_state,
        node_metrics: symmetrical_node_metrics(node_metrics),
    };

    vec![metrics, symmetrical_metrics]
}

fn symmetrical_node_metrics(
    metrics: &NodeMetrics<Action, Predictions, MovesLeftSnapshot>,
) -> NodeMetrics<Action, Predictions, MovesLeftSnapshot> {
    let children_symmetry = metrics
        .children()
        .iter()
        .map(|m| EdgeMetrics::new(m.action().horizontal_symmetry(), m.visits(), *m.snapshot()))
        .collect();

    NodeMetrics::new(
        metrics.predictions().clone(),
        metrics.visits(),
        children_symmetry,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::Value;
    use engine::GameState as GameStateTrait;

    fn get_symmetries_game_state(game_state: GameState) -> Vec<GameState> {
        let symmetries = get_symmetries(PositionMetrics {
            game_state,
            node_metrics: NodeMetrics::new(Predictions::new(Value::new(0.0, 0.0), 0.0), 0, vec![]),
        });

        symmetries.into_iter().map(|s| s.game_state).collect()
    }

    #[test]
    fn test_symmetry_initial_state() {
        let game_state: GameState = GameState::initial();
        let symmetries = get_symmetries_game_state(game_state);
        let original = symmetries.first().unwrap();
        let mirrored = symmetries.last().unwrap();

        assert_eq!(original.to_string(), mirrored.to_string());
    }

    #[test]
    fn test_symmetry_column_1_mirrors_to_column_7() {
        let game_state = GameState::initial().drop_piece(1);

        let symmetries = get_symmetries_game_state(game_state);
        let original = symmetries.first().unwrap();
        let mirrored = symmetries.last().unwrap();

        // A piece in column 1 should mirror to column 7
        assert!(
            original.p2_piece_board != mirrored.p2_piece_board
                || original.p1_piece_board != mirrored.p1_piece_board
        );

        // The mirror of the mirror should equal the original
        let double_mirrored = mirrored.horizontal_symmetry();
        assert_eq!(original.p1_piece_board, double_mirrored.p1_piece_board);
        assert_eq!(original.p2_piece_board, double_mirrored.p2_piece_board);
    }

    #[test]
    fn test_symmetry_column_4_is_unchanged() {
        let game_state = GameState::initial().drop_piece(4);

        let symmetries = get_symmetries_game_state(game_state);
        let original = symmetries.first().unwrap();
        let mirrored = symmetries.last().unwrap();

        assert_eq!(original.to_string(), mirrored.to_string());
    }

    #[test]
    fn test_action_symmetry_roundtrip() {
        for col in 1..=7usize {
            let action = Action::from(col);
            assert_eq!(action, action.horizontal_symmetry().horizontal_symmetry());
        }
    }

    #[test]
    fn test_action_symmetry_column_mapping() {
        assert_eq!(
            Action::from(1usize).horizontal_symmetry(),
            Action::from(7usize)
        );
        assert_eq!(
            Action::from(2usize).horizontal_symmetry(),
            Action::from(6usize)
        );
        assert_eq!(
            Action::from(3usize).horizontal_symmetry(),
            Action::from(5usize)
        );
        assert_eq!(
            Action::from(4usize).horizontal_symmetry(),
            Action::from(4usize)
        );
        assert_eq!(
            Action::from(5usize).horizontal_symmetry(),
            Action::from(3usize)
        );
        assert_eq!(
            Action::from(6usize).horizontal_symmetry(),
            Action::from(2usize)
        );
        assert_eq!(
            Action::from(7usize).horizontal_symmetry(),
            Action::from(1usize)
        );
    }
}
