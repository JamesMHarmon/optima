#[cfg(test)]
mod test {
    use arimaa::{Action, GameState, Value};
    use engine::GameState as GameStateTrait;
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

    #[test]
    fn test_deblunder() {
        let mut metrics = vec![PositionMetricsExtended {
            metrics: PositionMetrics {
                game_state: GameState::initial(),
                score: Value::new([0.0, 0.0]),
                policy: NodeMetrics {
                    visits: 10,
                    value: Value::new([0.0, 0.0]),
                    moves_left: 5.0,
                    children: vec![NodeChildMetrics::new("a1n".parse().unwrap(), 1.0, 0.5, 10)],
                },
                moves_left: 0,
            },
            chosen_action: "a1n".parse().unwrap(),
            move_number: 2,
        }];

        deblunder(&mut metrics, 0.1, 0.1);
    }
}
