#[cfg(test)]
#[allow(non_snake_case)]
mod test {
    use approx::assert_abs_diff_eq;
    use common::MovesLeftPropagatedValue;
    use arimaa::{Action, GameState, Predictions, Value};
    use engine::{GameState as GameStateTrait};
    use model::{EdgeMetrics, NodeMetrics, PositionMetrics};

    use crate::{
        arimaa_sampler::{ArimaaSampler, ArimaaPStore},
        sample::PositionMetricsExtended,
    };

    fn deblunder(
        metrics: &mut [PositionMetricsExtended<GameState, Action, Predictions, MovesLeftPropagatedValue>],
        q_diff_threshold: f32,
        q_diff_width: f32,
    ) {
        crate::deblunder::deblunder::<_, _, _, _, ArimaaPStore, ArimaaSampler>(
            metrics,
            q_diff_threshold,
            q_diff_width,
        );
    }

    fn game_state(is_player_one: bool) -> GameState {
        let mut game_state = GameState::initial();
        while game_state.is_p1_turn_to_move() != is_player_one {
            game_state = game_state.take_action(game_state.valid_actions().first().unwrap());
        }

        game_state
    }

    fn node_child<As>(action: As, Q: f32, M: f32, visits: usize) -> EdgeMetrics<Action, MovesLeftPropagatedValue>
    where
        As: AsRef<str>,
    {
        let action = action.as_ref().parse::<Action>().unwrap();
        let propagated_values = MovesLeftPropagatedValue::new(Q, M);
        EdgeMetrics::new(action, visits, propagated_values)
    }

    fn node_metrics(children: Vec<EdgeMetrics<Action, MovesLeftPropagatedValue>>) -> NodeMetrics<Action, Predictions, MovesLeftPropagatedValue> {
        let predictions = Predictions::new(Value::new([0.0, 0.0]), 0.0);

        NodeMetrics {
            visits: 0,
            predictions,
            children,
        }
    }

    fn position_metrics(
        is_player_one: bool,
        score: Value,
        moves_left: usize,
        move_number: usize,
        chosen_action: impl AsRef<str>,
        children: Vec<EdgeMetrics<Action, MovesLeftPropagatedValue>>,
    ) -> PositionMetricsExtended<GameState, Action, Value, MovesLeftPropagatedValue> {
        let target_score = Predictions::new(score, moves_left);

        PositionMetricsExtended {
            metrics: PositionMetrics {
                game_state: game_state(is_player_one),
                policy: node_metrics(children)
            },
            target_score,
            chosen_action: chosen_action.as_ref().parse().unwrap(),
            move_number,
        }
    }

    #[test]
    fn test_deblunder_no_changes_if_below_threshold() {
        let mut metrics = vec![
            position_metrics(
                true,
                Value::new([1.0, 0.0]),
                8,
                90,
                "a1n",
                vec![
                    node_child("a1n", 0.8, 25.0, 30),
                    node_child("a2n", 0.5, 25.0, 20),
                    node_child("a3n", 0.1, 25.0, 10),
                ],
            ),
            position_metrics(
                false,
                Value::new([1.0, 0.0]),
                7,
                91,
                "a1n",
                vec![
                    node_child("a1n", 0.8, 25.0, 30),
                    node_child("a2n", 0.5, 25.0, 20),
                    node_child("a3n", 0.1, 25.0, 10),
                ],
            ),
        ];

        deblunder(&mut metrics, 0.1, 0.1);

        assert_abs_diff_eq!(metrics[0].metrics.score.get_value_for_player(1), 1.0);
        assert_abs_diff_eq!(metrics[0].metrics.score.get_value_for_player(2), 0.0);
        assert_abs_diff_eq!(metrics[0].metrics.moves_left, 2);

        assert_abs_diff_eq!(metrics[1].metrics.score.get_value_for_player(1), 1.0);
        assert_abs_diff_eq!(metrics[1].metrics.score.get_value_for_player(2), 0.0);
        assert_abs_diff_eq!(metrics[1].metrics.moves_left, 1);
    }

    #[test]
    fn test_deblunder_sets_score_to_Q_if_blunder() {
        let mut metrics = vec![
            position_metrics(
                true,
                Value::new([1.0, 0.0]),
                10,
                90,
                "a1n",
                vec![
                    node_child("a1n", 0.6, 100.0, 30),
                    node_child("a2n", 0.5, 100.0, 20),
                    node_child("a3n", 0.1, 100.0, 10),
                ],
            ),
            position_metrics(
                false,
                Value::new([1.0, 0.0]),
                9,
                91,
                "a2n",
                vec![
                    node_child("a1n", 0.8, 105.0, 30),
                    node_child("a2n", 0.5, 100.0, 20),
                    node_child("a3n", 0.1, 100.0, 10),
                ],
            ),
            position_metrics(
                true,
                Value::new([1.0, 0.0]),
                8,
                92,
                "a1n",
                vec![
                    node_child("a1n", 0.8, 100.0, 30),
                    node_child("a2n", 0.5, 100.0, 20),
                    node_child("a3n", 0.1, 100.0, 10),
                ],
            ),
            position_metrics(
                true,
                Value::new([1.0, 0.0]),
                1,
                99,
                "a1n",
                vec![
                    node_child("a1n", 0.8, 100.0, 30),
                    node_child("a2n", 0.5, 100.0, 20),
                    node_child("a3n", 0.1, 100.0, 10),
                ],
            ),
        ];

        deblunder(&mut metrics, 0.1, 0.1);

        assert_abs_diff_eq!(metrics[0].metrics.score.get_value_for_player(1), 0.6);
        assert_abs_diff_eq!(metrics[0].metrics.moves_left, 15);

        assert_abs_diff_eq!(metrics[1].metrics.score.get_value_for_player(2), 0.8);
        assert_abs_diff_eq!(metrics[1].metrics.moves_left, 14);

        assert_abs_diff_eq!(metrics[2].metrics.score.get_value_for_player(1), 1.0);
        assert_abs_diff_eq!(metrics[2].metrics.moves_left, 8);
    }

    #[test]
    fn test_deblunder_sets_score_to_Q_if_blunder_with_width() {
        let mut metrics = vec![
            position_metrics(
                true,
                Value::new([1.0, 0.0]),
                10,
                90,
                "a1n",
                vec![
                    node_child("a1n", 0.6, 100.0, 30),
                    node_child("a2n", 0.5, 100.0, 20),
                    node_child("a3n", 0.1, 100.0, 10),
                ],
            ),
            position_metrics(
                false,
                Value::new([1.0, 0.0]),
                9,
                91,
                "a2n",
                vec![
                    node_child("a1n", 0.7, 100.0, 30),
                    node_child("a2n", 0.5, 100.0, 20),
                    node_child("a3n", 0.1, 100.0, 10),
                ],
            ),
            position_metrics(
                true,
                Value::new([1.0, 0.0]),
                8,
                92,
                "a1n",
                vec![
                    node_child("a1n", 0.8, 100.0, 30),
                    node_child("a2n", 0.5, 100.0, 20),
                    node_child("a3n", 0.1, 100.0, 10),
                ],
            ),
        ];

        deblunder(&mut metrics, 0.1, 0.2);

        assert_abs_diff_eq!(metrics[0].metrics.score.get_value_for_player(1), 0.8);

        assert_abs_diff_eq!(metrics[1].metrics.score.get_value_for_player(2), 0.35);

        assert_abs_diff_eq!(metrics[2].metrics.score.get_value_for_player(1), 1.0);
    }

    #[test]
    fn test_deblunder_multiple_blunders() {
        let mut metrics = vec![
            position_metrics(
                true,
                Value::new([1.0, 0.0]),
                10,
                90,
                "a1n",
                vec![
                    node_child("a1n", 0.65, 40.0, 30),
                    node_child("a2n", 0.5, 50.0, 20),
                    node_child("a3n", 0.1, 75.0, 10),
                ],
            ),
            position_metrics(
                false,
                Value::new([1.0, 0.0]),
                9,
                91,
                "a2n",
                vec![
                    node_child("a1n", 0.7, 40.0, 30),
                    node_child("a2n", 0.5, 100.0, 20),
                    node_child("a3n", 0.1, 100.0, 10),
                ],
            ),
            position_metrics(
                true,
                Value::new([1.0, 0.0]),
                8,
                92,
                "a1n",
                vec![
                    node_child("a1n", 0.8, 100.0, 30),
                    node_child("a2n", 0.5, 100.0, 20),
                    node_child("a3n", 0.1, 100.0, 10),
                ],
            ),
            position_metrics(
                false,
                Value::new([1.0, 0.0]),
                7,
                93,
                "a2n",
                vec![
                    node_child("a1n", 0.7, 100.0, 30),
                    node_child("a2n", 0.5, 100.0, 20),
                    node_child("a3n", 0.1, 100.0, 10),
                ],
            ),
            position_metrics(
                true,
                Value::new([1.0, 0.0]),
                6,
                94,
                "a1n",
                vec![
                    node_child("a1n", 0.8, 100.0, 30),
                    node_child("a2n", 0.5, 100.0, 20),
                    node_child("a3n", 0.1, 100.0, 10),
                ],
            ),
        ];

        deblunder(&mut metrics, 0.1, 0.2);

        assert_abs_diff_eq!(metrics[0].metrics.score.get_value_for_player(1), 0.775);
        assert_eq!(metrics[0].metrics.moves_left, 1);

        assert_abs_diff_eq!(metrics[1].metrics.score.get_value_for_player(2), 0.525);
        assert_eq!(metrics[1].metrics.moves_left, 1);

        assert_abs_diff_eq!(metrics[2].metrics.score.get_value_for_player(1), 0.9);
        assert_eq!(metrics[2].metrics.moves_left, 8);

        assert_abs_diff_eq!(metrics[3].metrics.score.get_value_for_player(2), 0.35);
        assert_eq!(metrics[3].metrics.moves_left, 7);

        assert_abs_diff_eq!(metrics[4].metrics.score.get_value_for_player(1), 1.0);
        assert_eq!(metrics[4].metrics.moves_left, 1);
    }

    #[test]
    fn test_disable_deblunder_when_threshold_is_zero() {
        let mut metrics = vec![
            position_metrics(
                true,
                Value::new([1.0, 0.0]),
                10,
                90,
                "a1n",
                vec![
                    node_child("a1n", 0.65, 100.0, 30),
                    node_child("a2n", 0.5, 100.0, 20),
                    node_child("a3n", 0.1, 100.0, 10),
                ],
            ),
            position_metrics(
                false,
                Value::new([1.0, 0.0]),
                9,
                91,
                "a2n",
                vec![
                    node_child("a1n", 0.7, 100.0, 30),
                    node_child("a2n", 0.5, 100.0, 20),
                    node_child("a3n", 0.1, 100.0, 10),
                ],
            ),
            position_metrics(
                true,
                Value::new([1.0, 0.0]),
                8,
                92,
                "a1n",
                vec![
                    node_child("a1n", 0.8, 100.0, 30),
                    node_child("a2n", 0.5, 100.0, 20),
                    node_child("a3n", 0.1, 100.0, 10),
                ],
            ),
            position_metrics(
                false,
                Value::new([1.0, 0.0]),
                7,
                93,
                "a2n",
                vec![
                    node_child("a1n", 0.7, 100.0, 30),
                    node_child("a2n", 0.5, 100.0, 20),
                    node_child("a3n", 0.1, 100.0, 10),
                ],
            ),
            position_metrics(
                true,
                Value::new([1.0, 0.0]),
                6,
                94,
                "a1n",
                vec![
                    node_child("a1n", 0.8, 100.0, 30),
                    node_child("a2n", 0.5, 100.0, 20),
                    node_child("a3n", 0.1, 100.0, 10),
                ],
            ),
        ];

        deblunder(&mut metrics, 0.0, 0.2);

        assert_abs_diff_eq!(metrics[0].metrics.score.get_value_for_player(1), 1.0);
        assert_eq!(metrics[0].metrics.moves_left, 10);

        assert_abs_diff_eq!(metrics[1].metrics.score.get_value_for_player(2), 0.0);
        assert_eq!(metrics[1].metrics.moves_left, 9);

        assert_abs_diff_eq!(metrics[2].metrics.score.get_value_for_player(1), 1.0);
        assert_eq!(metrics[2].metrics.moves_left, 8);

        assert_abs_diff_eq!(metrics[3].metrics.score.get_value_for_player(2), 0.0);
        assert_eq!(metrics[3].metrics.moves_left, 7);

        assert_abs_diff_eq!(metrics[4].metrics.score.get_value_for_player(1), 1.0);
        assert_eq!(metrics[4].metrics.moves_left, 6);
    }
}
