#[cfg(test)]
#[allow(non_snake_case)]
mod test {
    use approx::assert_abs_diff_eq;
    use arimaa::{Action, GameState, Predictions, Value};
    use common::{GameLength, PlayerValue};
    use engine::GameState as GameStateTrait;
    use model::{EdgeMetrics, NodeMetrics, PositionMetrics};
    use puct::MovesLeftSnapshot;

    use crate::{
        arimaa_sampler::{ArimaaPStore, ArimaaSampler},
        sample::PositionMetricsExtended,
    };

    type Metric = PositionMetricsExtended<GameState, Action, Predictions, MovesLeftSnapshot>;
    type MetricNode = NodeMetrics<Action, Predictions, MovesLeftSnapshot>;
    type MetricEdge = EdgeMetrics<Action, MovesLeftSnapshot>;
    type MetricPosition = PositionMetrics<GameState, Action, Predictions, MovesLeftSnapshot>;

    fn deblunder(metrics: &mut [Metric], q_diff_threshold: f32, q_diff_width: f32) {
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

    fn node_child<As>(action: As, Q: f32, M: f32, visits: usize) -> MetricEdge
    where
        As: AsRef<str>,
    {
        let action = action.as_ref().parse::<Action>().unwrap();
        let snapshot = MovesLeftSnapshot {
            p1_sum: Q as f64,
            p2_sum: Q as f64,
            game_length_sum: M as f64,
            total_weight: 1,
        };
        EdgeMetrics::new(action, visits, snapshot)
    }

    fn node_child_p1_p2<As>(action: As, p1_Q: f32, p2_Q: f32, M: f32, visits: usize) -> MetricEdge
    where
        As: AsRef<str>,
    {
        let action = action.as_ref().parse::<Action>().unwrap();
        let snapshot = MovesLeftSnapshot {
            p1_sum: p1_Q as f64,
            p2_sum: p2_Q as f64,
            game_length_sum: M as f64,
            total_weight: 1,
        };
        EdgeMetrics::new(action, visits, snapshot)
    }

    fn node_metrics(children: Vec<MetricEdge>) -> MetricNode {
        let predictions = Predictions::new(Value::new(0.0, 0.0), 0.0);

        NodeMetrics {
            visits: 0,
            predictions,
            children,
        }
    }

    fn position_metrics(
        is_player_one: bool,
        score: Value,
        game_length: f32,
        chosen_action: impl AsRef<str>,
        children: Vec<MetricEdge>,
    ) -> Metric {
        let target_score = Predictions::new(score, game_length);

        PositionMetricsExtended {
            metrics: MetricPosition {
                game_state: game_state(is_player_one),
                node_metrics: node_metrics(children),
            },
            target_score,
            chosen_action: chosen_action.as_ref().parse().unwrap(),
        }
    }

    #[test]
    fn test_deblunder_no_changes_if_below_threshold() {
        let mut metrics = vec![
            position_metrics(
                true,
                Value::new(1.0, 0.0),
                98.0,
                "a1n",
                vec![
                    node_child("a1n", 0.8, 25.0, 30),
                    node_child("a2n", 0.5, 25.0, 20),
                    node_child("a3n", 0.1, 25.0, 10),
                ],
            ),
            position_metrics(
                false,
                Value::new(1.0, 0.0),
                98.0,
                "a1n",
                vec![
                    node_child("a1n", 0.8, 25.0, 30),
                    node_child("a2n", 0.5, 25.0, 20),
                    node_child("a3n", 0.1, 25.0, 10),
                ],
            ),
        ];

        deblunder(&mut metrics, 0.1, 0.1);

        assert_abs_diff_eq!(metrics[0].target_score.value().player_value(1), 1.0);
        assert_abs_diff_eq!(metrics[0].target_score.value().player_value(2), 0.0);
        assert_abs_diff_eq!(metrics[0].target_score.game_length(), 98.0);

        assert_abs_diff_eq!(metrics[1].target_score.value().player_value(1), 1.0);
        assert_abs_diff_eq!(metrics[1].target_score.value().player_value(2), 0.0);
        assert_abs_diff_eq!(metrics[1].target_score.game_length(), 98.0);
    }

    #[test]
    fn test_deblunder_sets_score_to_Q_if_blunder() {
        let mut metrics = vec![
            position_metrics(
                true,
                Value::new(1.0, 0.0),
                100.0,
                "a1n",
                vec![
                    node_child("a1n", 0.6, 100.0, 30),
                    node_child("a2n", 0.5, 100.0, 20),
                    node_child("a3n", 0.1, 100.0, 10),
                ],
            ),
            position_metrics(
                false,
                Value::new(1.0, 0.0),
                100.0,
                "a2n",
                vec![
                    node_child("a1n", 0.8, 105.0, 30),
                    node_child("a2n", 0.5, 100.0, 20),
                    node_child("a3n", 0.1, 100.0, 10),
                ],
            ),
            position_metrics(
                true,
                Value::new(1.0, 0.0),
                100.0,
                "a1n",
                vec![
                    node_child("a1n", 0.8, 100.0, 30),
                    node_child("a2n", 0.5, 100.0, 20),
                    node_child("a3n", 0.1, 100.0, 10),
                ],
            ),
            position_metrics(
                true,
                Value::new(1.0, 0.0),
                100.0,
                "a1n",
                vec![
                    node_child("a1n", 0.8, 100.0, 30),
                    node_child("a2n", 0.5, 100.0, 20),
                    node_child("a3n", 0.1, 100.0, 10),
                ],
            ),
        ];

        deblunder(&mut metrics, 0.1, 0.1);

        assert_abs_diff_eq!(metrics[0].target_score.value().player_value(1), 0.6);
        assert_abs_diff_eq!(metrics[0].target_score.game_length(), 105.0);

        assert_abs_diff_eq!(metrics[1].target_score.value().player_value(2), 0.8);
        assert_abs_diff_eq!(metrics[1].target_score.game_length(), 105.0);

        assert_abs_diff_eq!(metrics[2].target_score.value().player_value(1), 1.0);
        assert_abs_diff_eq!(metrics[2].target_score.game_length(), 100.0);
    }

    #[test]
    fn test_deblunder_sets_score_to_Q_if_blunder_with_width() {
        let mut metrics = vec![
            position_metrics(
                true,
                Value::new(1.0, 0.0),
                100.0,
                "a1n",
                vec![
                    node_child("a1n", 0.6, 100.0, 30),
                    node_child("a2n", 0.5, 100.0, 20),
                    node_child("a3n", 0.1, 100.0, 10),
                ],
            ),
            position_metrics(
                false,
                Value::new(1.0, 0.0),
                100.0,
                "a2n",
                vec![
                    node_child("a1n", 0.7, 100.0, 30),
                    node_child("a2n", 0.5, 100.0, 20),
                    node_child("a3n", 0.1, 100.0, 10),
                ],
            ),
            position_metrics(
                true,
                Value::new(1.0, 0.0),
                100.0,
                "a1n",
                vec![
                    node_child("a1n", 0.8, 100.0, 30),
                    node_child("a2n", 0.5, 100.0, 20),
                    node_child("a3n", 0.1, 100.0, 10),
                ],
            ),
        ];

        deblunder(&mut metrics, 0.1, 0.2);

        assert_abs_diff_eq!(metrics[0].target_score.value().player_value(1), 0.8);

        assert_abs_diff_eq!(metrics[1].target_score.value().player_value(2), 0.35);

        assert_abs_diff_eq!(metrics[2].target_score.value().player_value(1), 1.0);
    }

    #[test]
    fn test_deblunder_multiple_blunders() {
        let mut metrics = vec![
            position_metrics(
                true,
                Value::new(1.0, 0.0),
                100.0,
                "a1n",
                vec![
                    node_child("a1n", 0.65, 40.0, 30),
                    node_child("a2n", 0.5, 50.0, 20),
                    node_child("a3n", 0.1, 75.0, 10),
                ],
            ),
            position_metrics(
                false,
                Value::new(1.0, 0.0),
                100.0,
                "a2n",
                vec![
                    node_child("a1n", 0.7, 40.0, 30),
                    node_child("a2n", 0.5, 100.0, 20),
                    node_child("a3n", 0.1, 100.0, 10),
                ],
            ),
            position_metrics(
                true,
                Value::new(1.0, 0.0),
                100.0,
                "a1n",
                vec![
                    node_child("a1n", 0.8, 100.0, 30),
                    node_child("a2n", 0.5, 100.0, 20),
                    node_child("a3n", 0.1, 100.0, 10),
                ],
            ),
            position_metrics(
                false,
                Value::new(1.0, 0.0),
                100.0,
                "a2n",
                vec![
                    node_child("a1n", 0.7, 100.0, 30),
                    node_child("a2n", 0.5, 100.0, 20),
                    node_child("a3n", 0.1, 100.0, 10),
                ],
            ),
            position_metrics(
                true,
                Value::new(1.0, 0.0),
                100.0,
                "a1n",
                vec![
                    node_child("a1n", 0.8, 100.0, 30),
                    node_child("a2n", 0.5, 100.0, 20),
                    node_child("a3n", 0.1, 100.0, 10),
                ],
            ),
        ];

        deblunder(&mut metrics, 0.1, 0.2);

        assert_abs_diff_eq!(metrics[0].target_score.value().player_value(1), 0.775);
        assert_abs_diff_eq!(metrics[0].target_score.game_length(), 70.00001);

        assert_abs_diff_eq!(metrics[1].target_score.value().player_value(2), 0.525);
        assert_abs_diff_eq!(metrics[1].target_score.game_length(), 70.00001);

        assert_abs_diff_eq!(metrics[2].target_score.value().player_value(1), 0.9);
        assert_eq!(metrics[2].target_score.game_length(), 100.0);

        assert_abs_diff_eq!(metrics[3].target_score.value().player_value(2), 0.35);
        assert_abs_diff_eq!(metrics[3].target_score.game_length(), 100.0);

        assert_abs_diff_eq!(metrics[4].target_score.value().player_value(1), 1.0);
        assert_abs_diff_eq!(metrics[4].target_score.game_length(), 100.0);
    }

    #[test]
    fn test_disable_deblunder_when_threshold_is_zero() {
        let mut metrics = vec![
            position_metrics(
                true,
                Value::new(1.0, 0.0),
                10.0,
                "a1n",
                vec![
                    node_child("a1n", 0.65, 100.0, 30),
                    node_child("a2n", 0.5, 100.0, 20),
                    node_child("a3n", 0.1, 100.0, 10),
                ],
            ),
            position_metrics(
                false,
                Value::new(1.0, 0.0),
                9.0,
                "a2n",
                vec![
                    node_child("a1n", 0.7, 100.0, 30),
                    node_child("a2n", 0.5, 100.0, 20),
                    node_child("a3n", 0.1, 100.0, 10),
                ],
            ),
            position_metrics(
                true,
                Value::new(1.0, 0.0),
                8.0,
                "a1n",
                vec![
                    node_child("a1n", 0.8, 100.0, 30),
                    node_child("a2n", 0.5, 100.0, 20),
                    node_child("a3n", 0.1, 100.0, 10),
                ],
            ),
            position_metrics(
                false,
                Value::new(1.0, 0.0),
                7.0,
                "a2n",
                vec![
                    node_child("a1n", 0.7, 100.0, 30),
                    node_child("a2n", 0.5, 100.0, 20),
                    node_child("a3n", 0.1, 100.0, 10),
                ],
            ),
            position_metrics(
                true,
                Value::new(1.0, 0.0),
                6.0,
                "a1n",
                vec![
                    node_child("a1n", 0.8, 100.0, 30),
                    node_child("a2n", 0.5, 100.0, 20),
                    node_child("a3n", 0.1, 100.0, 10),
                ],
            ),
        ];

        deblunder(&mut metrics, 0.0, 0.2);

        assert_abs_diff_eq!(metrics[0].target_score.value().player_value(1), 1.0);
        assert_eq!(metrics[0].target_score.game_length(), 10.0);

        assert_abs_diff_eq!(metrics[1].target_score.value().player_value(2), 0.0);
        assert_eq!(metrics[1].target_score.game_length(), 9.0);

        assert_abs_diff_eq!(metrics[2].target_score.value().player_value(1), 1.0);
        assert_eq!(metrics[2].target_score.game_length(), 8.0);

        assert_abs_diff_eq!(metrics[3].target_score.value().player_value(2), 0.0);
        assert_eq!(metrics[3].target_score.game_length(), 7.0);

        assert_abs_diff_eq!(metrics[4].target_score.value().player_value(1), 1.0);
        assert_eq!(metrics[4].target_score.game_length(), 6.0);
    }

    #[test]
    fn test_deblunder_triggers_on_exact_threshold_but_noop_when_q_mix_is_zero() {
        // q_diff == threshold => q_mix_amt = 0.0, so the pushed stack frame should not change
        // the predictions (but still exercises the >= boundary).
        let mut metrics = vec![position_metrics(
            true,
            Value::new(0.9, 0.1),
            33.0,
            "a2n",
            vec![
                node_child("a1n", 0.6, 80.0, 30),
                node_child("a2n", 0.5, 10.0, 20),
                node_child("a3n", 0.1, 10.0, 10),
            ],
        )];

        deblunder(&mut metrics, 0.1, 0.5);

        assert_abs_diff_eq!(metrics[0].target_score.value().player_value(1), 0.9);
        assert_abs_diff_eq!(metrics[0].target_score.value().player_value(2), 0.1);
        assert_abs_diff_eq!(metrics[0].target_score.game_length(), 33.0);
    }

    #[test]
    fn test_deblunder_saturates_q_mix_at_one_and_updates_only_player_to_move() {
        // Large q_diff relative to width => q_mix_amt clamped to 1.0.
        // For ArimaaSampler::mix_q this fully replaces ONLY the player_to_move value + game_length.
        let mut metrics = vec![position_metrics(
            true,
            Value::new(0.2, 0.9),
            10.0,
            "a2n",
            vec![
                node_child_p1_p2("a1n", 0.8, 0.3, 50.0, 40), // max visits
                node_child_p1_p2("a2n", 0.1, 0.7, 11.0, 10),
            ],
        )];

        deblunder(&mut metrics, 0.1, 0.1);

        assert_abs_diff_eq!(metrics[0].target_score.value().player_value(1), 0.8);
        assert_abs_diff_eq!(metrics[0].target_score.value().player_value(2), 0.9);
        assert_abs_diff_eq!(metrics[0].target_score.game_length(), 50.0);
    }

    #[test]
    fn test_deblunder_does_not_trigger_when_chosen_action_is_max_visits_child() {
        let mut metrics = vec![position_metrics(
            true,
            Value::new(0.3, 0.4),
            12.0,
            "a1n",
            vec![
                node_child_p1_p2("a1n", 0.6, 0.1, 77.0, 30), // max visits + chosen
                node_child_p1_p2("a2n", 0.9, 0.9, 5.0, 10),
            ],
        )];

        deblunder(&mut metrics, 0.1, 0.2);

        assert_abs_diff_eq!(metrics[0].target_score.value().player_value(1), 0.3);
        assert_abs_diff_eq!(metrics[0].target_score.value().player_value(2), 0.4);
        assert_abs_diff_eq!(metrics[0].target_score.game_length(), 12.0);
    }

    #[test]
    fn test_deblunder_does_not_trigger_when_chosen_has_higher_q_than_max_visits() {
        // Deblunder compares chosen Q to max-visits Q (not max-Q), so negative q_diff should not
        // trigger.
        let mut metrics = vec![position_metrics(
            true,
            Value::new(0.55, 0.45),
            22.0,
            "a2n",
            vec![
                node_child("a1n", 0.4, 100.0, 30), // max visits
                node_child("a2n", 0.9, 5.0, 10),
            ],
        )];

        deblunder(&mut metrics, 0.1, 0.2);

        assert_abs_diff_eq!(metrics[0].target_score.value().player_value(1), 0.55);
        assert_abs_diff_eq!(metrics[0].target_score.value().player_value(2), 0.45);
        assert_abs_diff_eq!(metrics[0].target_score.game_length(), 22.0);
    }

    #[test]
    fn test_deblunder_width_zero_behaves_like_full_mix_when_above_threshold() {
        let mut metrics = vec![position_metrics(
            false,
            Value::new(0.1, 0.2),
            9.0,
            "a2n",
            vec![
                node_child_p1_p2("a1n", 0.9, 0.75, 123.0, 50), // max visits
                node_child_p1_p2("a2n", 0.9, 0.10, 7.0, 10),
            ],
        )];

        // player 2 to move => uses p2_Q. q_diff = 0.75 - 0.10 = 0.65 > 0.1.
        deblunder(&mut metrics, 0.1, 0.0);

        assert_abs_diff_eq!(metrics[0].target_score.value().player_value(1), 0.1);
        assert_abs_diff_eq!(metrics[0].target_score.value().player_value(2), 0.75);
        assert_abs_diff_eq!(metrics[0].target_score.game_length(), 123.0);
    }

    #[test]
    #[should_panic(expected = "Specified action was not found")]
    fn test_deblunder_panics_if_chosen_action_missing_from_children() {
        let mut metrics = vec![position_metrics(
            true,
            Value::new(0.2, 0.8),
            1.0,
            "b1n",
            vec![node_child("a1n", 0.6, 1.0, 10)],
        )];

        deblunder(&mut metrics, 0.1, 0.2);
    }

    #[test]
    #[should_panic]
    fn test_deblunder_panics_if_no_children() {
        let mut metrics = vec![position_metrics(
            true,
            Value::new(0.2, 0.8),
            1.0,
            "a1n",
            vec![],
        )];

        deblunder(&mut metrics, 0.1, 0.2);
    }

    #[test]
    #[should_panic(expected = "Q mix must be between 0.0 and 1.0")]
    fn test_deblunder_panics_for_negative_width_when_blunder_triggers() {
        let mut metrics = vec![position_metrics(
            true,
            Value::new(0.2, 0.9),
            10.0,
            "a2n",
            vec![
                node_child("a1n", 0.8, 50.0, 40), // max visits
                node_child("a2n", 0.1, 10.0, 10),
            ],
        )];

        // width < 0 => q_mix_amt becomes negative, which should trip QMix's invariant.
        deblunder(&mut metrics, 0.1, -0.2);
    }

    #[test]
    fn test_deblunder_partial_mix_blends_value_and_game_length() {
        // Choose q_diff such that q_mix_amt = 0.5.
        // q_mix_amt = (q_diff - threshold) / width
        // => q_diff = threshold + 0.5 * width
        // threshold=0.1, width=0.2 => q_diff=0.2
        // Use max_visits_q=0.8 and chosen_q=0.6 => q_diff=0.2
        let mut metrics = vec![position_metrics(
            true,
            Value::new(0.2, 0.9),
            10.0,
            "a2n",
            vec![
                node_child_p1_p2("a1n", 0.8, 0.3, 50.0, 40), // max visits
                node_child_p1_p2("a2n", 0.6, 0.7, 11.0, 10),
            ],
        )];

        deblunder(&mut metrics, 0.1, 0.2);

        // mixed_value = 0.5*post + 0.5*pre = 0.5*(0.2+0.8)=0.5
        // mixed_gl = 0.5*(10+50)=30
        assert_abs_diff_eq!(metrics[0].target_score.value().player_value(1), 0.5);
        assert_abs_diff_eq!(metrics[0].target_score.value().player_value(2), 0.9);
        assert_abs_diff_eq!(metrics[0].target_score.game_length(), 30.0, epsilon = 1e-4);
    }

    #[test]
    fn test_deblunder_negative_width_does_not_panic_if_no_blunder() {
        let mut metrics = vec![position_metrics(
            true,
            Value::new(0.3, 0.4),
            12.0,
            "a2n",
            vec![
                node_child("a1n", 0.8, 99.0, 40), // max visits
                node_child("a2n", 0.75, 1.0, 10),
            ],
        )];

        // q_diff = 0.05 < threshold => should not compute q_mix at all.
        deblunder(&mut metrics, 0.1, -0.2);

        assert_abs_diff_eq!(metrics[0].target_score.value().player_value(1), 0.3);
        assert_abs_diff_eq!(metrics[0].target_score.value().player_value(2), 0.4);
        assert_abs_diff_eq!(metrics[0].target_score.game_length(), 12.0);
    }

    #[test]
    fn test_deblunder_noop_on_empty_metrics() {
        let mut metrics: Vec<Metric> = vec![];
        deblunder(&mut metrics, 0.1, 0.2);
        assert!(metrics.is_empty());
    }

    #[test]
    fn test_deblunder_propagates_latest_prediction_backward_for_same_player() {
        // Two p1-to-move positions, latest (end) is a blunder.
        // Earlier position should receive the latest prediction even if it isn't itself a blunder.
        let mut metrics = vec![
            position_metrics(
                true,
                Value::new(0.7, 0.8),
                20.0,
                "a1n",
                vec![
                    node_child("a1n", 0.8, 20.0, 30),
                    node_child("a2n", 0.7, 20.0, 10),
                ],
            ),
            position_metrics(
                true,
                Value::new(0.2, 0.4),
                10.0,
                "a2n",
                vec![
                    node_child("a1n", 0.9, 100.0, 40), // max visits
                    node_child("a2n", 0.0, 10.0, 10),
                ],
            ),
        ];

        deblunder(&mut metrics, 0.1, 0.1);

        // Latest position fully mixes to pre-blunder snapshot.
        assert_abs_diff_eq!(metrics[1].target_score.value().player_value(1), 0.9);
        assert_abs_diff_eq!(metrics[1].target_score.value().player_value(2), 0.4);
        assert_abs_diff_eq!(metrics[1].target_score.game_length(), 100.0);

        // Earlier position inherits the latest p1 prediction from the prediction stack.
        assert_abs_diff_eq!(metrics[0].target_score.value().player_value(1), 0.9);
        assert_abs_diff_eq!(metrics[0].target_score.value().player_value(2), 0.4);
        assert_abs_diff_eq!(metrics[0].target_score.game_length(), 100.0);
    }

    #[test]
    fn test_deblunder_uses_player_to_move_for_q_diff_arimaa() {
        // Construct a position where only P2's Q shows a blunder.
        // If q_diff used P1's Q by mistake, it would not trigger.
        let mut metrics = vec![position_metrics(
            false,
            Value::new(0.4, 0.2),
            9.0,
            "a2n",
            vec![
                // max visits: p1_Q low, p2_Q high
                node_child_p1_p2("a1n", 0.1, 0.9, 123.0, 50),
                // chosen: p2_Q low => blunder for P2 only
                node_child_p1_p2("a2n", 0.9, 0.1, 7.0, 10),
            ],
        )];

        deblunder(&mut metrics, 0.1, 0.1);

        // P2 to move => only P2 value updates.
        assert_abs_diff_eq!(metrics[0].target_score.value().player_value(1), 0.4);
        assert_abs_diff_eq!(metrics[0].target_score.value().player_value(2), 0.9);
        assert_abs_diff_eq!(metrics[0].target_score.game_length(), 123.0);
    }

    #[cfg(feature = "quoridor")]
    mod quoridor_deblunder {
        use super::*;

        use common::VictoryMargin;
        use puct::VictoryMarginSnapshot;
        use quoridor::{
            Action as QAction, GameState as QGameState, Predictions as QPredictions,
            Value as QValue,
        };

        use crate::quoridor_sampler::{QuoridorSampler, QuoridorVStore};

        type QMetric =
            PositionMetricsExtended<QGameState, QAction, QPredictions, VictoryMarginSnapshot>;
        type QMetricNode = NodeMetrics<QAction, QPredictions, VictoryMarginSnapshot>;
        type QMetricEdge = EdgeMetrics<QAction, VictoryMarginSnapshot>;
        type QMetricPosition =
            PositionMetrics<QGameState, QAction, QPredictions, VictoryMarginSnapshot>;

        fn deblunder_q(metrics: &mut [QMetric], q_diff_threshold: f32, q_diff_width: f32) {
            crate::deblunder::deblunder::<_, _, _, _, QuoridorVStore, QuoridorSampler>(
                metrics,
                q_diff_threshold,
                q_diff_width,
            );
        }

        fn q_game_state(is_player_one: bool) -> QGameState {
            let mut game_state = QGameState::initial();

            while game_state.p1_turn_to_move() != is_player_one {
                let action = game_state
                    .valid_actions()
                    .next()
                    .expect("Quoridor initial state should have valid actions");
                game_state.take_action(&action);
            }

            game_state
        }

        fn q_node_child<As>(
            action: As,
            p1_Q: f32,
            p2_Q: f32,
            victory_margin: f32,
            game_length: f32,
            visits: usize,
        ) -> QMetricEdge
        where
            As: AsRef<str>,
        {
            let action = action.as_ref().parse::<QAction>().unwrap();
            let snapshot = VictoryMarginSnapshot {
                p1_sum: p1_Q as f64,
                p2_sum: p2_Q as f64,
                victory_margin_sum: victory_margin as f64,
                game_length_sum: game_length as f64,
                total_weight: 1,
            };
            EdgeMetrics::new(action, visits, snapshot)
        }

        fn q_node_metrics(children: Vec<QMetricEdge>) -> QMetricNode {
            let predictions = QPredictions::new(QValue::new(0.0, 0.0), 0.0, 0.0);

            NodeMetrics {
                visits: 0,
                predictions,
                children,
            }
        }

        fn q_position_metrics(
            is_player_one: bool,
            score: QValue,
            victory_margin: f32,
            game_length: f32,
            chosen_action: impl AsRef<str>,
            children: Vec<QMetricEdge>,
        ) -> QMetric {
            let target_score = QPredictions::new(score, victory_margin, game_length);

            PositionMetricsExtended {
                metrics: QMetricPosition {
                    game_state: q_game_state(is_player_one),
                    node_metrics: q_node_metrics(children),
                },
                target_score,
                chosen_action: chosen_action.as_ref().parse().unwrap(),
            }
        }

        #[test]
        fn test_quoridor_deblunder_full_mix_updates_player_to_move_and_extras() {
            let mut metrics = vec![q_position_metrics(
                true,
                QValue::new(0.2, 0.9),
                1.0,
                10.0,
                "e2",
                vec![
                    q_node_child("e3", 0.8, 0.1, 5.0, 100.0, 40), // max visits
                    q_node_child("e2", 0.0, 0.2, 0.0, 11.0, 10),
                ],
            )];

            deblunder_q(&mut metrics, 0.1, 0.1);

            // P1 to move: only P1 value changes; victory_margin and game_length fully replaced.
            assert_abs_diff_eq!(metrics[0].target_score.value().player_value(1), 0.8);
            assert_abs_diff_eq!(metrics[0].target_score.value().player_value(2), 0.9);
            assert_abs_diff_eq!(metrics[0].target_score.victory_margin(), 5.0);
            assert_abs_diff_eq!(metrics[0].target_score.game_length(), 100.0);
        }

        #[test]
        fn test_quoridor_deblunder_partial_mix_blends_all_mixed_fields() {
            // Make q_mix_amt = 0.5 by using q_diff=0.2 with threshold=0.1 and width=0.2.
            let mut metrics = vec![q_position_metrics(
                true,
                QValue::new(0.2, 0.9),
                1.0,
                10.0,
                "e2",
                vec![
                    q_node_child("e3", 0.8, 0.1, 5.0, 50.0, 40), // max visits
                    q_node_child("e2", 0.6, 0.2, 0.0, 11.0, 10),
                ],
            )];

            deblunder_q(&mut metrics, 0.1, 0.2);

            // value(p1) = 0.5*(0.2 + 0.8) = 0.5
            assert_abs_diff_eq!(metrics[0].target_score.value().player_value(1), 0.5);
            assert_abs_diff_eq!(metrics[0].target_score.value().player_value(2), 0.9);
            // victory_margin = 0.5*(1.0 + 5.0) = 3.0
            assert_abs_diff_eq!(
                metrics[0].target_score.victory_margin(),
                3.0,
                epsilon = 1e-4
            );
            // game_length = 0.5*(10.0 + 50.0) = 30.0
            assert_abs_diff_eq!(metrics[0].target_score.game_length(), 30.0, epsilon = 1e-4);
        }

        #[test]
        fn test_quoridor_deblunder_uses_player_to_move_for_q_diff_and_mixing() {
            // P2 to move; only P2's Q indicates a blunder.
            // If q_diff incorrectly used P1's Q, q_diff would be negative and no blunder.
            let mut metrics = vec![q_position_metrics(
                false,
                QValue::new(0.4, 0.2),
                2.0,
                9.0,
                "e2",
                vec![
                    // max visits: p1_Q low, p2_Q high
                    q_node_child("e3", 0.1, 0.9, 8.0, 123.0, 50),
                    // chosen: p2_Q low
                    q_node_child("e2", 0.9, 0.1, 0.0, 7.0, 10),
                ],
            )];

            deblunder_q(&mut metrics, 0.1, 0.1);

            // P2 to move: only P2 value changes; victory_margin + game_length replaced.
            assert_abs_diff_eq!(metrics[0].target_score.value().player_value(1), 0.4);
            assert_abs_diff_eq!(metrics[0].target_score.value().player_value(2), 0.9);
            assert_abs_diff_eq!(metrics[0].target_score.victory_margin(), 8.0);
            assert_abs_diff_eq!(metrics[0].target_score.game_length(), 123.0);
        }
    }
}
