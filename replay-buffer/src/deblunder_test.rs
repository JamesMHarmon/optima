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

        NodeMetrics::new(predictions, 0, children)
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

    #[test]
    fn test_deblunder_width_zero_at_exact_threshold_uses_full_mix() {
        // When q_diff_width == 0.0 and q_diff == q_diff_threshold exactly:
        //   (q_diff - threshold) / width = 0.0 / 0.0 = NaN
        //   NaN.min(1.0) = 1.0  (Rust's f32::min uses LLVM minnum: returns non-NaN operand)
        // So q_mix_amt = 1.0 — full replacement — consistent with the step-function intent
        // that any q_diff >= threshold with width=0.0 produces full mix.
        //
        // Use exact f32 fractions (powers of 2) so that (Q_a1n - Q_a2n) == threshold exactly
        // in IEEE 754: 0.75 - 0.25 = 0.5 = threshold with zero rounding error.
        let mut metrics = vec![position_metrics(
            true,
            Value::new(0.2, 0.8),
            10.0,
            "a2n",
            vec![
                node_child("a1n", 0.75, 50.0, 30), // q_diff = 0.75-0.25 = 0.5 == threshold
                node_child("a2n", 0.25, 20.0, 10),
            ],
        )];

        deblunder(&mut metrics, 0.5, 0.0);

        // q_mix_amt = 1.0 (full replacement), not 0.0.
        assert_abs_diff_eq!(metrics[0].target_score.value().player_value(1), 0.75);
        assert_abs_diff_eq!(metrics[0].target_score.value().player_value(2), 0.8); // unchanged
        assert_abs_diff_eq!(metrics[0].target_score.game_length(), 50.0);
    }

    #[test]
    fn test_deblunder_consecutive_same_player_blunders() {
        // Two P1 positions in a row (no P2 between them), both blunders.
        // Frame stack grows twice for P1. Each blunder frame uses the max_visits snapshot
        // at its own position, compounding corrections through the stack.
        //
        // metrics[1] (processed first in reverse): q_diff = 0.65-0.2 = 0.45 → q_mix = 1.0
        //   frame0.p1 = (1.0,0.0)@50; frame1.p1 = mix(frame0.p1, a1n@m1, 1.0):
        //   = (1-1)*1.0 + 1*0.65 = 0.65, gl=(1-1)*50+1*40=40 → frame1=(0.65,0.0)@40
        //
        // metrics[0] (processed second): q_diff = 0.7-0.4 = 0.3 → q_mix = 1.0
        //   frame1.p1 already set; push frame2.p1 = mix(frame1.p1, a1n@m0, 1.0):
        //   = (1-1)*0.65 + 1*0.7 = 0.7, gl=(1-1)*40+1*60=60 → frame2=(0.7,0.0)@60
        let mut metrics = vec![
            position_metrics(
                true,
                Value::new(1.0, 0.0),
                50.0,
                "a2n",
                vec![
                    node_child("a1n", 0.7, 60.0, 30),
                    node_child("a2n", 0.4, 50.0, 10),
                ],
            ),
            position_metrics(
                true,
                Value::new(1.0, 0.0),
                50.0,
                "a2n",
                vec![
                    node_child("a1n", 0.65, 40.0, 30),
                    node_child("a2n", 0.2, 50.0, 10),
                ],
            ),
        ];

        deblunder(&mut metrics, 0.1, 0.2);

        // metrics[1] (later in game, first processed): corrected by its own blunder.
        assert_abs_diff_eq!(metrics[1].target_score.value().player_value(1), 0.65);
        assert_abs_diff_eq!(metrics[1].target_score.value().player_value(2), 0.0);
        assert_abs_diff_eq!(metrics[1].target_score.game_length(), 40.0);

        // metrics[0] (earlier in game, second processed): uses its own max_visits snapshot
        // mixed against the already-corrected frame1 prediction, not the original target.
        assert_abs_diff_eq!(metrics[0].target_score.value().player_value(1), 0.7);
        assert_abs_diff_eq!(metrics[0].target_score.value().player_value(2), 0.0);
        assert_abs_diff_eq!(metrics[0].target_score.game_length(), 60.0);
    }

    #[test]
    fn test_deblunder_p1_blunder_corrects_earlier_p2_position() {
        // P1 blunders at metrics[1]. P2 has a position at metrics[0] (before the blunder).
        // A blunder pushes a new prediction frame for ALL players, so P2's earlier position
        // must also receive a corrected prediction. The correction uses P2's own max_visits
        // snapshot at that position, mixed with P2's existing prediction via the blunder
        // frame's q_mix_amt.
        //
        // Processing reverse:
        // metrics[1] (p1): q_diff = 0.9-0.3 = 0.6 → q_mix=1.0
        //   frame1.p1 = mix(frame0.p1=(0.8,0.2)@20, a1n@m1, 1.0):
        //     p1 mixed_value = 0.9, mixed_gl = 30 → frame1=(0.9,0.2)@30
        //
        // metrics[0] (p2): no blunder (chosen=max_visits=a1n)
        //   set_initial: frame0.p2 = Value(0.1,0.7); game_length already 20 (from p1 set_initial)
        //   set_if_not fills frame1.p2 = mix(frame0.p2=(0.1,0.7)@20, a1n@m0, 1.0):
        //     p2 mixed_value = 0.6, mixed_gl computation = 14.0 but frame1.gl already 30 (p1's)
        //     → stored as Value(0.1,0.6) in frame1; gl remains 30.0
        //   latest(p2) → frame1 → Predictions(Value(0.1,0.6), 30.0)
        let mut metrics = vec![
            position_metrics(
                false, // P2 to move
                Value::new(0.1, 0.7),
                15.0,
                "a1n", // chosen = max visits → no blunder for P2
                vec![
                    node_child_p1_p2("a1n", 0.3, 0.6, 14.0, 30),
                    node_child_p1_p2("a2n", 0.8, 0.2, 30.0, 10),
                ],
            ),
            position_metrics(
                true, // P1 to move
                Value::new(0.8, 0.2),
                20.0,
                "a2n", // P1 chose poorly; max_visits=a1n has much higher P1 Q
                vec![
                    node_child_p1_p2("a1n", 0.9, 0.1, 30.0, 40), // max visits
                    node_child_p1_p2("a2n", 0.3, 0.7, 20.0, 10),
                ],
            ),
        ];

        deblunder(&mut metrics, 0.1, 0.1);

        // metrics[1]: P1's value replaced with max-visits Q=0.9; P2 unchanged; gl from a1n=30.
        assert_abs_diff_eq!(metrics[1].target_score.value().player_value(1), 0.9);
        assert_abs_diff_eq!(metrics[1].target_score.value().player_value(2), 0.2);
        assert_abs_diff_eq!(metrics[1].target_score.game_length(), 30.0);

        // metrics[0]: P2's value is corrected toward P2's max_visits Q (0.6);
        // game_length is inherited from P1's blunder frame (30.0), not P2's own max_visits (14.0).
        assert_abs_diff_eq!(metrics[0].target_score.value().player_value(1), 0.1);
        assert_abs_diff_eq!(metrics[0].target_score.value().player_value(2), 0.6);
        assert_abs_diff_eq!(metrics[0].target_score.game_length(), 30.0);
    }

    #[test]
    fn test_deblunder_equal_visits_last_child_wins_tie_no_blunder() {
        // child_max_visits() uses max_by_key(|c| c.visits), which returns the LAST element
        // among ties. When the chosen action is the last child and ties for max visits,
        // it becomes the max-visits child → q_diff = 0 → no blunder, even though another
        // child has higher Q and the same visit count.
        let mut metrics = vec![position_metrics(
            true,
            Value::new(1.0, 0.0),
            10.0,
            "a2n", // chosen = last child in list
            vec![
                node_child("a1n", 0.8, 10.0, 30), // higher Q, same visits as a2n
                node_child("a2n", 0.3, 10.0, 30), // lower Q, same visits — LAST → wins tie
            ],
        )];

        deblunder(&mut metrics, 0.1, 0.2);

        // a2n wins tie (last in list) → max_visits_child = a2n = chosen → q_diff = 0.
        assert_abs_diff_eq!(metrics[0].target_score.value().player_value(1), 1.0);
        assert_abs_diff_eq!(metrics[0].target_score.game_length(), 10.0);
    }

    #[test]
    fn test_deblunder_equal_visits_last_child_wins_tie_blunder_fires() {
        // Mirror of the above: the chosen action is the FIRST child; the second child
        // (same visits, last in list) wins the tie and becomes max_visits_child.
        // Now q_diff = second.Q - first.Q > threshold → blunder fires.
        let mut metrics = vec![position_metrics(
            true,
            Value::new(1.0, 0.0),
            10.0,
            "a2n", // chosen = first child in list
            vec![
                node_child("a2n", 0.3, 10.0, 30), // chosen, same visits — FIRST
                node_child("a1n", 0.8, 40.0, 30), // higher Q, same visits — LAST → wins tie
            ],
        )];

        deblunder(&mut metrics, 0.1, 0.2);

        // a1n wins tie → q_diff = 0.8-0.3 = 0.5 → q_mix_amt = (0.5-0.1)/0.2 = 2.0 → 1.0
        assert_abs_diff_eq!(metrics[0].target_score.value().player_value(1), 0.8);
        assert_abs_diff_eq!(metrics[0].target_score.game_length(), 40.0);
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

            NodeMetrics::new(predictions, 0, children)
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
