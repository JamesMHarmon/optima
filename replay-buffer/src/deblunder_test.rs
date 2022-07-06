#[cfg(test)]
#[allow(non_snake_case)]
mod test {
    use approx::assert_abs_diff_eq;
    use std::fmt::Debug;
    use std::str::FromStr;

    use arimaa::{Action, GameState, Value};
    use engine::{GameState as GameStateTrait, Value as ValueTrait};
    use model::{NodeChildMetrics, NodeMetrics, PositionMetrics};

    use crate::{
        arimaa_sampler::{ArimaaSampler, ArimaaVStore},
        sample::PositionMetricsExtended,
    };

    fn deblunder(
        metrics: &mut [PositionMetricsExtended<GameState, Action, Value>],
        q_diff_threshold: f32,
        q_diff_width: f32,
    ) {
        crate::deblunder::deblunder::<_, _, _, ArimaaVStore, ArimaaSampler>(
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

    fn node_child<A, As>(action: As, Q: f32, M: f32, visits: usize) -> NodeChildMetrics<A>
    where
        A: FromStr,
        A::Err: Debug,
        As: AsRef<str>,
    {
        let action = action.as_ref().parse::<A>().unwrap();
        NodeChildMetrics::new(action, Q, M, visits)
    }

    fn node_metrics<A>(children: Vec<NodeChildMetrics<A>>) -> NodeMetrics<A, Value> {
        NodeMetrics {
            visits: 0,
            value: Value::new([0.0, 0.0]),
            moves_left: 0.0,
            children,
        }
    }

    fn position_metrics(
        is_player_one: bool,
        score: Value,
        moves_left: usize,
        chosen_action: impl AsRef<str>,
        children: Vec<NodeChildMetrics<Action>>,
    ) -> PositionMetricsExtended<GameState, Action, Value> {
        PositionMetricsExtended {
            metrics: PositionMetrics {
                game_state: game_state(is_player_one),
                score,
                policy: node_metrics(children),
                moves_left,
            },
            chosen_action: chosen_action.as_ref().parse().unwrap(),
            move_number: 100 - moves_left,
        }
    }

    #[test]
    fn test_deblunder_no_changes_if_below_threshold() {
        let mut metrics = vec![
            position_metrics(
                true,
                Value::new([1.0, 0.0]),
                10,
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
                9,
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

        assert_abs_diff_eq!(metrics[1].metrics.score.get_value_for_player(1), 1.0);
        assert_abs_diff_eq!(metrics[1].metrics.score.get_value_for_player(2), 0.0);
    }

    #[test]
    fn test_deblunder_sets_score_to_Q_if_blunder() {
        let mut metrics = vec![
            position_metrics(
                true,
                Value::new([1.0, 0.0]),
                10,
                "a1n",
                vec![
                    node_child("a1n", 0.6, 25.0, 30),
                    node_child("a2n", 0.5, 25.0, 20),
                    node_child("a3n", 0.1, 25.0, 10),
                ],
            ),
            position_metrics(
                false,
                Value::new([1.0, 0.0]),
                9,
                "a2n",
                vec![
                    node_child("a1n", 0.8, 25.0, 30),
                    node_child("a2n", 0.5, 25.0, 20),
                    node_child("a3n", 0.1, 25.0, 10),
                ],
            ),
            position_metrics(
                true,
                Value::new([1.0, 0.0]),
                8,
                "a1n",
                vec![
                    node_child("a1n", 0.8, 25.0, 30),
                    node_child("a2n", 0.5, 25.0, 20),
                    node_child("a3n", 0.1, 25.0, 10),
                ],
            ),
        ];

        deblunder(&mut metrics, 0.1, 0.1);

        assert_abs_diff_eq!(metrics[0].metrics.score.get_value_for_player(1), 0.6);

        assert_abs_diff_eq!(metrics[1].metrics.score.get_value_for_player(2), 0.8);

        assert_abs_diff_eq!(metrics[2].metrics.score.get_value_for_player(1), 1.0);
    }

    #[test]
    fn test_deblunder_sets_score_to_Q_if_blunder_with_width() {
        let mut metrics = vec![
            position_metrics(
                true,
                Value::new([1.0, 0.0]),
                10,
                "a1n",
                vec![
                    node_child("a1n", 0.6, 25.0, 30),
                    node_child("a2n", 0.5, 25.0, 20),
                    node_child("a3n", 0.1, 25.0, 10),
                ],
            ),
            position_metrics(
                false,
                Value::new([1.0, 0.0]),
                9,
                "a2n",
                vec![
                    node_child("a1n", 0.7, 25.0, 30),
                    node_child("a2n", 0.5, 25.0, 20),
                    node_child("a3n", 0.1, 25.0, 10),
                ],
            ),
            position_metrics(
                true,
                Value::new([1.0, 0.0]),
                8,
                "a1n",
                vec![
                    node_child("a1n", 0.8, 25.0, 30),
                    node_child("a2n", 0.5, 25.0, 20),
                    node_child("a3n", 0.1, 25.0, 10),
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
                "a2n",
                vec![
                    node_child("a1n", 0.7, 40.0, 30),
                    node_child("a2n", 0.5, 25.0, 20),
                    node_child("a3n", 0.1, 25.0, 10),
                ],
            ),
            position_metrics(
                true,
                Value::new([1.0, 0.0]),
                8,
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
                "a2n",
                vec![
                    node_child("a1n", 0.7, 25.0, 30),
                    node_child("a2n", 0.5, 25.0, 20),
                    node_child("a3n", 0.1, 25.0, 10),
                ],
            ),
            position_metrics(
                true,
                Value::new([1.0, 0.0]),
                6,
                "a1n",
                vec![
                    node_child("a1n", 0.8, 25.0, 30),
                    node_child("a2n", 0.5, 25.0, 20),
                    node_child("a3n", 0.1, 25.0, 10),
                ],
            ),
        ];

        deblunder(&mut metrics, 0.1, 0.2);

        assert_abs_diff_eq!(metrics[0].metrics.score.get_value_for_player(1), 0.775);
        assert_eq!(metrics[0].metrics.moves_left, 42);

        assert_abs_diff_eq!(metrics[1].metrics.score.get_value_for_player(2), 0.525);
        assert_eq!(metrics[1].metrics.moves_left, 41);

        assert_abs_diff_eq!(metrics[2].metrics.score.get_value_for_player(1), 0.9);
        assert_eq!(metrics[2].metrics.moves_left, 27);

        assert_abs_diff_eq!(metrics[3].metrics.score.get_value_for_player(2), 0.35);
        assert_eq!(metrics[3].metrics.moves_left, 26);

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
                "a1n",
                vec![
                    node_child("a1n", 0.65, 25.0, 30),
                    node_child("a2n", 0.5, 25.0, 20),
                    node_child("a3n", 0.1, 25.0, 10),
                ],
            ),
            position_metrics(
                false,
                Value::new([1.0, 0.0]),
                9,
                "a2n",
                vec![
                    node_child("a1n", 0.7, 25.0, 30),
                    node_child("a2n", 0.5, 25.0, 20),
                    node_child("a3n", 0.1, 25.0, 10),
                ],
            ),
            position_metrics(
                true,
                Value::new([1.0, 0.0]),
                8,
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
                "a2n",
                vec![
                    node_child("a1n", 0.7, 25.0, 30),
                    node_child("a2n", 0.5, 25.0, 20),
                    node_child("a3n", 0.1, 25.0, 10),
                ],
            ),
            position_metrics(
                true,
                Value::new([1.0, 0.0]),
                6,
                "a1n",
                vec![
                    node_child("a1n", 0.8, 25.0, 30),
                    node_child("a2n", 0.5, 25.0, 20),
                    node_child("a3n", 0.1, 25.0, 10),
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
