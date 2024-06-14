use crate::{counting_game::CountingGameState, Temperature, CPUCT};

struct CPUCTTest {
    cpuct: f32,
}

impl CPUCT for CPUCTTest {
    type State = CountingGameState;

    fn cpuct(&self, _: &Self::State, _: usize, _: bool) -> f32 {
        self.cpuct
    }
}

struct TempTest;

impl Temperature for TempTest {
    type State = CountingGameState;

    fn temp(&self, _: &Self::State) -> f32 {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        MovesLeftBackpropagationStrategy, MovesLeftSelectionStrategy,
        MovesLeftStrategyOptions,
    };

    use super::super::MCTS;

    use super::super::counting_game::{
        CountingAction, CountingAnalyzer, CountingGameEngine, CountingGamePredictions,
        CountingGameState,
    };
    use super::{CPUCTTest, TempTest};
    use assert_approx_eq::assert_approx_eq;
    use common::MovesLeftPropagatedValue;
    use engine::GameState;
    use model::{EdgeMetrics, NodeMetrics};

    const ERROR_DIFF: f32 = 0.02;
    const ERROR_DIFF_W: f32 = 0.01;

    fn assert_metrics(
        left: &NodeMetrics<CountingAction, CountingGamePredictions, MovesLeftPropagatedValue>,
        right: &NodeMetrics<CountingAction, CountingGamePredictions, MovesLeftPropagatedValue>,
    ) {
        assert_eq!(left.visits, right.visits);
        assert_eq!(left.children.len(), right.children.len());

        for (left, right) in left.children.iter().zip(right.children.iter()) {
            assert_eq!(left.action(), right.action());
            assert_approx_eq!(left.avg_value(), right.avg_value(), ERROR_DIFF_W);
            let max_visits = left.visits().max(right.visits());
            let allowed_diff = (max_visits as f32) * ERROR_DIFF + 0.9;
            assert_approx_eq!(left.visits() as f32, right.visits() as f32, allowed_diff);
        }
    }

    fn edge_metrics(
        action: CountingAction,
        visits: usize,
        value: f32,
    ) -> EdgeMetrics<CountingAction, MovesLeftPropagatedValue> {
        EdgeMetrics::new(
            action,
            visits,
            MovesLeftPropagatedValue::new(value * visits as f32, 0.0),
        )
    }

    #[tokio::test]
    async fn test_mcts_is_deterministic() {
        let game_state = CountingGameState::initial();
        let game_engine = CountingGameEngine::new();
        let analyzer = CountingAnalyzer::new([0.3, 0.3, 0.4]);
        let backpropagation_strategy = MovesLeftBackpropagationStrategy::new(&game_engine);
        let options = MovesLeftStrategyOptions::new(0.0, 0.0, 0.0, 1.0, 10.0, 0.05);
        let selection_strategy = MovesLeftSelectionStrategy::new(CPUCTTest { cpuct: 3.0 }, options);
        let temp = TempTest;

        let mut mcts = MCTS::new(
            game_state.to_owned(),
            &game_engine,
            &analyzer,
            &backpropagation_strategy,
            &selection_strategy,
            temp,
            1,
        );

        let temp = TempTest;
        let mut mcts2 = MCTS::new(
            game_state,
            &game_engine,
            &analyzer,
            &backpropagation_strategy,
            &selection_strategy,
            temp,
            1,
        );

        mcts.search_visits(800).await.unwrap();
        mcts2.search_visits(800).await.unwrap();

        let metrics = mcts.get_root_node_metrics().unwrap();
        let metrics2 = mcts.get_root_node_metrics().unwrap();

        assert_metrics(&metrics, &metrics2);
    }

    #[tokio::test]
    async fn test_mcts_chooses_best_p1_move() {
        let game_state = CountingGameState::initial();
        let game_engine = CountingGameEngine::new();
        let analyzer = CountingAnalyzer::new([0.3, 0.3, 0.4]);
        let backpropagation_strategy = MovesLeftBackpropagationStrategy::new(&game_engine);
        let options = MovesLeftStrategyOptions::new(0.0, 0.0, 0.0, 1.0, 10.0, 0.05);
        let selection_strategy = MovesLeftSelectionStrategy::new(CPUCTTest { cpuct: 1.0 }, options);
        let temp = TempTest;

        let mut mcts = MCTS::new(
            game_state,
            &game_engine,
            &analyzer,
            &backpropagation_strategy,
            &selection_strategy,
            temp,
            1,
        );

        mcts.search_visits(800).await.unwrap();

        let action = mcts.select_action().unwrap();

        assert_eq!(action, CountingAction::Increment);
    }

    #[tokio::test]
    async fn test_mcts_chooses_best_p2_move() {
        let game_state = CountingGameState::initial();
        let game_engine = CountingGameEngine::new();
        let analyzer = CountingAnalyzer::new([0.3, 0.3, 0.4]);
        let backpropagation_strategy = MovesLeftBackpropagationStrategy::new(&game_engine);
        let options = MovesLeftStrategyOptions::new(0.0, 0.0, 0.0, 1.0, 10.0, 0.05);
        let selection_strategy = MovesLeftSelectionStrategy::new(CPUCTTest { cpuct: 1.0 }, options);
        let temp = TempTest;

        let mut mcts = MCTS::new(
            game_state,
            &game_engine,
            &analyzer,
            &backpropagation_strategy,
            &selection_strategy,
            temp,
            1,
        );

        mcts.advance_to_action(CountingAction::Increment)
            .await
            .unwrap();

        mcts.search_visits(800).await.unwrap();
        let action = mcts.select_action().unwrap();

        assert_eq!(action, CountingAction::Decrement);
    }

    #[tokio::test]
    async fn test_mcts_should_overcome_policy_through_value() {
        let game_state = CountingGameState::initial();
        let game_engine = CountingGameEngine::new();
        let analyzer = CountingAnalyzer::new([0.3, 0.3, 0.4]);
        let backpropagation_strategy = MovesLeftBackpropagationStrategy::new(&game_engine);
        let options = MovesLeftStrategyOptions::new(0.0, 0.0, 0.0, 1.0, 10.0, 0.05);
        let selection_strategy = MovesLeftSelectionStrategy::new(CPUCTTest { cpuct: 2.5 }, options);
        let temp = TempTest;

        let mut mcts = MCTS::new(
            game_state,
            &game_engine,
            &analyzer,
            &backpropagation_strategy,
            &selection_strategy,
            temp,
            1,
        );

        mcts.advance_to_action(CountingAction::Increment)
            .await
            .unwrap();

        mcts.search_visits(800).await.unwrap();
        let details = mcts.get_focus_node_details().unwrap().unwrap();
        let top_edge = details.children.first().unwrap();

        assert_eq!(top_edge.action, CountingAction::Stay);

        mcts.search_visits(8000).await.unwrap();
        let action = mcts.select_action().unwrap();

        assert_eq!(action, CountingAction::Decrement);
    }

    #[tokio::test]
    async fn test_mcts_advance_to_next_works_without_search() {
        let game_state = CountingGameState::initial();
        let game_engine = CountingGameEngine::new();
        let analyzer: CountingAnalyzer = CountingAnalyzer::new([0.3, 0.3, 0.4]);
        let backpropagation_strategy: MovesLeftBackpropagationStrategy<
            CountingGameEngine,
            CountingGameState,
            CountingAction,
            CountingGamePredictions,
        > = MovesLeftBackpropagationStrategy::new(&game_engine);
        let options = MovesLeftStrategyOptions::new(0.0, 0.0, 0.0, 1.0, 10.0, 0.05);
        let selection_strategy: MovesLeftSelectionStrategy<
            CountingGameState,
            CountingAction,
            CountingGamePredictions,
            CPUCTTest,
        > = MovesLeftSelectionStrategy::new(CPUCTTest { cpuct: 3.0 }, options);
        let temp = TempTest;

        let mut mcts: MCTS<_, _, _, _, _, _, _, _, MovesLeftPropagatedValue> = MCTS::new(
            game_state,
            &game_engine,
            &analyzer,
            &backpropagation_strategy,
            &selection_strategy,
            temp,
            1,
        );

        mcts.advance_to_action(CountingAction::Increment)
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn test_mcts_metrics_returns_accurate_results() {
        let game_state = CountingGameState::initial();
        let game_engine = CountingGameEngine::new();
        let analyzer = CountingAnalyzer::new([0.3, 0.3, 0.4]);
        let backpropagation_strategy = MovesLeftBackpropagationStrategy::new(&game_engine);
        let options = MovesLeftStrategyOptions::new(0.0, 0.0, 0.0, 1.0, 10.0, 0.05);
        let selection_strategy = MovesLeftSelectionStrategy::new(CPUCTTest { cpuct: 1.0 }, options);
        let temp = TempTest;

        let mut mcts = MCTS::new(
            game_state,
            &game_engine,
            &analyzer,
            &backpropagation_strategy,
            &selection_strategy,
            temp,
            1,
        );

        mcts.search_visits(800).await.unwrap();

        let metrics = mcts.get_root_node_metrics().unwrap();

        assert_metrics(
            &metrics,
            &NodeMetrics {
                visits: 800,
                predictions: CountingGamePredictions([0.0, 0.0]),
                children: vec![
                    edge_metrics(CountingAction::Stay, 304, 0.5),
                    edge_metrics(CountingAction::Decrement, 177, 0.49),
                    edge_metrics(CountingAction::Increment, 312, 0.509),
                ],
            },
        );
    }

    #[tokio::test]
    async fn test_mcts_weights_policy_initially() {
        let game_state = CountingGameState::initial();
        let game_engine = CountingGameEngine::new();
        let analyzer = CountingAnalyzer::new([0.3, 0.3, 0.4]);
        let backpropagation_strategy = MovesLeftBackpropagationStrategy::new(&game_engine);
        let options = MovesLeftStrategyOptions::new(0.0, 0.0, 0.0, 1.0, 10.0, 0.05);
        let selection_strategy = MovesLeftSelectionStrategy::new(CPUCTTest { cpuct: 3.0 }, options);
        let temp = TempTest;

        let mut mcts = MCTS::new(
            game_state,
            &game_engine,
            &analyzer,
            &backpropagation_strategy,
            &selection_strategy,
            temp,
            1,
        );

        mcts.search_visits(100).await.unwrap();

        let metrics = mcts.get_root_node_metrics().unwrap();

        assert_metrics(
            &metrics,
            &NodeMetrics {
                visits: 100,
                predictions: CountingGamePredictions([0.0, 0.0]),
                children: vec![
                    edge_metrics(CountingAction::Stay, 40, 0.5),
                    edge_metrics(CountingAction::Decrement, 29, 0.49),
                    edge_metrics(CountingAction::Increment, 31, 0.51),
                ],
            },
        );
    }

    #[tokio::test]
    async fn test_mcts_works_with_single_node() {
        let game_state = CountingGameState::initial();
        let game_engine = CountingGameEngine::new();
        let analyzer = CountingAnalyzer::new([0.3, 0.3, 0.4]);
        let backpropagation_strategy = MovesLeftBackpropagationStrategy::new(&game_engine);
        let options = MovesLeftStrategyOptions::new(0.0, 0.0, 0.0, 1.0, 10.0, 0.05);
        let selection_strategy = MovesLeftSelectionStrategy::new(CPUCTTest { cpuct: 3.0 }, options);
        let temp = TempTest;

        let mut mcts = MCTS::new(
            game_state,
            &game_engine,
            &analyzer,
            &backpropagation_strategy,
            &selection_strategy,
            temp,
            1,
        );

        mcts.search_visits(1).await.unwrap();

        let metrics = mcts.get_root_node_metrics().unwrap();

        assert_metrics(
            &metrics,
            &NodeMetrics {
                visits: 1,
                predictions: CountingGamePredictions([0.0, 0.0]),
                children: vec![
                    edge_metrics(CountingAction::Increment, 0, 0.0),
                    edge_metrics(CountingAction::Decrement, 0, 0.0),
                    edge_metrics(CountingAction::Stay, 0, 0.0),
                ],
            },
        );
    }

    #[tokio::test]
    async fn test_mcts_works_with_two_nodes() {
        let game_state = CountingGameState::initial();
        let game_engine = CountingGameEngine::new();
        let analyzer = CountingAnalyzer::new([0.3, 0.3, 0.4]);
        let backpropagation_strategy = MovesLeftBackpropagationStrategy::new(&game_engine);
        let options = MovesLeftStrategyOptions::new(0.0, 0.0, 0.0, 1.0, 10.0, 0.05);
        let selection_strategy = MovesLeftSelectionStrategy::new(CPUCTTest { cpuct: 3.0 }, options);
        let temp = TempTest;

        let mut mcts = MCTS::new(
            game_state,
            &game_engine,
            &analyzer,
            &backpropagation_strategy,
            &selection_strategy,
            temp,
            1,
        );

        mcts.search_visits(2).await.unwrap();

        let metrics = mcts.get_root_node_metrics().unwrap();

        assert_metrics(
            &metrics,
            &NodeMetrics {
                visits: 2,
                predictions: CountingGamePredictions([0.0, 0.0]),
                children: vec![
                    edge_metrics(CountingAction::Stay, 1, 0.5),
                    edge_metrics(CountingAction::Increment, 0, 0.0),
                    edge_metrics(CountingAction::Decrement, 0, 0.0),
                ],
            },
        );
    }

    #[tokio::test]
    async fn test_mcts_clear_nodes_results_in_same_outcome() {
        let game_state = CountingGameState::initial();
        let game_engine = CountingGameEngine::new();
        let analyzer = CountingAnalyzer::new([0.3, 0.3, 0.4]);
        let backpropagation_strategy = MovesLeftBackpropagationStrategy::new(&game_engine);
        let options = MovesLeftStrategyOptions::new(0.0, 0.0, 0.0, 1.0, 10.0, 0.05);
        let selection_strategy = MovesLeftSelectionStrategy::new(CPUCTTest { cpuct: 3.0 }, options);
        let temp = || TempTest;
        let search_num_visits = 800;

        let mut non_clear_mcts = MCTS::new(
            game_state.clone(),
            &game_engine,
            &analyzer,
            &backpropagation_strategy,
            &selection_strategy,
            temp(),
            1,
        );

        non_clear_mcts
            .search_visits(search_num_visits)
            .await
            .unwrap();
        let action = non_clear_mcts.select_action().unwrap();
        non_clear_mcts
            .advance_to_action_retain(action)
            .await
            .unwrap();
        non_clear_mcts
            .search_visits(search_num_visits)
            .await
            .unwrap();

        let non_clear_metrics = non_clear_mcts.get_root_node_metrics().unwrap();

        let mut clear_mcts = MCTS::new(
            game_state.clone(),
            &game_engine,
            &analyzer,
            &backpropagation_strategy,
            &selection_strategy,
            temp(),
            1,
        );

        clear_mcts.search_visits(search_num_visits).await.unwrap();
        let action = clear_mcts.select_action().unwrap();
        clear_mcts.advance_to_action(action.clone()).await.unwrap();
        clear_mcts.search_visits(search_num_visits).await.unwrap();

        let clear_metrics = clear_mcts.get_root_node_metrics().unwrap();

        let mut initial_mcts = MCTS::new(
            game_state,
            &game_engine,
            &analyzer,
            &backpropagation_strategy,
            &selection_strategy,
            temp(),
            1,
        );

        initial_mcts.advance_to_action(action).await.unwrap();
        initial_mcts.search_visits(search_num_visits).await.unwrap();

        let initial_metrics = initial_mcts.get_root_node_metrics().unwrap();

        assert_metrics(&initial_metrics, &clear_metrics);
        assert_metrics(&non_clear_metrics, &clear_metrics);
    }
}
