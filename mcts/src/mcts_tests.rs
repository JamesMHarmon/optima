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
    use super::super::{MCTSOptions, MCTS};

    use super::super::counting_game::{
        CountingAction, CountingAnalyzer, CountingGameEngine, CountingGameState, Value,
    };
    use super::{CPUCTTest, TempTest};
    use assert_approx_eq::assert_approx_eq;
    use engine::GameState;
    use model::{EdgeMetrics, NodeMetrics};

    const ERROR_DIFF: f32 = 0.02;
    const ERROR_DIFF_W: f32 = 0.01;

    fn assert_metrics(
        left: &NodeMetrics<CountingAction, Value>,
        right: &NodeMetrics<CountingAction, Value>,
    ) {
        assert_eq!(left.visits, right.visits);
        assert_eq!(left.children.len(), right.children.len());

        for (left, right) in left.children.iter().zip(right.children.iter()) {
            assert_eq!(left.action(), right.action());
            assert_approx_eq!(left.Q(), right.Q(), ERROR_DIFF_W);
            let max_visits = left.visits().max(right.visits());
            let allowed_diff = (max_visits as f32) * ERROR_DIFF + 0.9;
            assert_approx_eq!(left.visits() as f32, right.visits() as f32, allowed_diff);
        }
    }

    #[tokio::test]
    async fn test_mcts_is_deterministic() {
        let game_state = CountingGameState::initial();
        let game_engine = CountingGameEngine::new();
        let analyzer = CountingAnalyzer::new();
        let cpuct = CPUCTTest { cpuct: 3.0 };
        let temp = TempTest;

        let mut mcts = MCTS::new(
            game_state.to_owned(),
            &game_engine,
            &analyzer,
            MCTSOptions::new(None, 0.0, 0.0, 0.0, 1.0, 10.0, 0.05, 1),
            cpuct,
            temp,
        );

        let cpuct = CPUCTTest { cpuct: 3.0 };
        let temp = TempTest;
        let mut mcts2 = MCTS::new(
            game_state,
            &game_engine,
            &analyzer,
            MCTSOptions::new(None, 0.0, 0.0, 0.0, 1.0, 10.0, 0.05, 1),
            cpuct,
            temp,
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
        let analyzer = CountingAnalyzer::new();
        let cpuct = CPUCTTest { cpuct: 1.0 };
        let temp = TempTest;

        let mut mcts = MCTS::new(
            game_state,
            &game_engine,
            &analyzer,
            MCTSOptions::new(None, 0.0, 0.0, 0.0, 1.0, 10.0, 0.05, 1),
            cpuct,
            temp,
        );

        mcts.search_visits(800).await.unwrap();
        let action = mcts.select_action().unwrap();

        assert_eq!(action, CountingAction::Increment);
    }

    #[tokio::test]
    async fn test_mcts_chooses_best_p2_move() {
        let game_state = CountingGameState::initial();
        let game_engine = CountingGameEngine::new();
        let analyzer = CountingAnalyzer::new();
        let cpuct = CPUCTTest { cpuct: 1.0 };
        let temp = TempTest;

        let mut mcts = MCTS::new(
            game_state,
            &game_engine,
            &analyzer,
            MCTSOptions::new(None, 0.0, 0.0, 0.0, 1.0, 10.0, 0.05, 1),
            cpuct,
            temp,
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
        let analyzer = CountingAnalyzer::new();
        let cpuct = CPUCTTest { cpuct: 2.5 };
        let temp = TempTest;

        let mut mcts = MCTS::new(
            game_state,
            &game_engine,
            &analyzer,
            MCTSOptions::new(None, 0.0, 0.0, 0.0, 1.0, 10.0, 0.05, 1),
            cpuct,
            temp,
        );

        mcts.advance_to_action(CountingAction::Increment)
            .await
            .unwrap();

        mcts.search_visits(800).await.unwrap();
        let details = mcts.get_focus_node_details().unwrap().unwrap();
        let (action, _) = details.children.first().unwrap();

        assert_eq!(*action, CountingAction::Stay);

        mcts.search_visits(8000).await.unwrap();
        let action = mcts.select_action().unwrap();

        assert_eq!(action, CountingAction::Decrement);
    }

    #[tokio::test]
    async fn test_mcts_advance_to_next_works_without_search() {
        let game_state = CountingGameState::initial();
        let game_engine = CountingGameEngine::new();
        let analyzer = CountingAnalyzer::new();
        let cpuct = CPUCTTest { cpuct: 3.0 };
        let temp = TempTest;

        let mut mcts = MCTS::new(
            game_state,
            &game_engine,
            &analyzer,
            MCTSOptions::new(None, 0.0, 0.0, 0.0, 1.0, 10.0, 0.05, 1),
            cpuct,
            temp,
        );

        mcts.advance_to_action(CountingAction::Increment)
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn test_mcts_metrics_returns_accurate_results() {
        let game_state = CountingGameState::initial();
        let game_engine = CountingGameEngine::new();
        let analyzer = CountingAnalyzer::new();
        let cpuct = CPUCTTest { cpuct: 1.0 };
        let temp = TempTest;

        let mut mcts = MCTS::new(
            game_state,
            &game_engine,
            &analyzer,
            MCTSOptions::new(None, 0.0, 0.0, 0.0, 1.0, 10.0, 0.05, 1),
            cpuct,
            temp,
        );

        mcts.search_visits(800).await.unwrap();

        let metrics = mcts.get_root_node_metrics().unwrap();

        assert_metrics(
            &metrics,
            &NodeMetrics {
                visits: 800,
                value: Value([0.0, 0.0]),
                moves_left: 0.0,
                children: vec![
                    EdgeMetrics::new(CountingAction::Stay, 0.5, 0.0, 304),
                    EdgeMetrics::new(CountingAction::Decrement, 0.49, 0.0, 177),
                    EdgeMetrics::new(CountingAction::Increment, 0.509, 0.0, 312),
                ],
            },
        );
    }

    #[tokio::test]
    async fn test_mcts_weights_policy_initially() {
        let game_state = CountingGameState::initial();
        let game_engine = CountingGameEngine::new();
        let analyzer = CountingAnalyzer::new();
        let cpuct = CPUCTTest { cpuct: 3.0 };
        let temp = TempTest;

        let mut mcts = MCTS::new(
            game_state,
            &game_engine,
            &analyzer,
            MCTSOptions::new(None, 0.0, 0.0, 0.0, 1.0, 10.0, 0.05, 1),
            cpuct,
            temp,
        );

        mcts.search_visits(100).await.unwrap();

        let metrics = mcts.get_root_node_metrics().unwrap();

        assert_metrics(
            &metrics,
            &NodeMetrics {
                visits: 100,
                value: Value([0.0, 0.0]),
                moves_left: 0.0,
                children: vec![
                    EdgeMetrics::new(CountingAction::Stay, 0.5, 0.0, 40),
                    EdgeMetrics::new(CountingAction::Decrement, 0.49, 0.0, 29),
                    EdgeMetrics::new(CountingAction::Increment, 0.51, 0.0, 31),
                ],
            },
        );
    }

    #[tokio::test]
    async fn test_mcts_works_with_single_node() {
        let game_state = CountingGameState::initial();
        let game_engine = CountingGameEngine::new();
        let analyzer = CountingAnalyzer::new();
        let cpuct = CPUCTTest { cpuct: 3.0 };
        let temp = TempTest;

        let mut mcts = MCTS::new(
            game_state,
            &game_engine,
            &analyzer,
            MCTSOptions::new(None, 0.0, 0.0, 0.0, 1.0, 10.0, 0.05, 1),
            cpuct,
            temp,
        );

        mcts.search_visits(1).await.unwrap();

        let metrics = mcts.get_root_node_metrics().unwrap();

        assert_metrics(
            &metrics,
            &NodeMetrics {
                visits: 1,
                value: Value([0.0, 0.0]),
                moves_left: 0.0,
                children: vec![
                    EdgeMetrics::new(CountingAction::Increment, 0.0, 0.0, 0),
                    EdgeMetrics::new(CountingAction::Decrement, 0.0, 0.0, 0),
                    EdgeMetrics::new(CountingAction::Stay, 0.0, 0.0, 0),
                ],
            },
        );
    }

    #[tokio::test]
    async fn test_mcts_works_with_two_nodes() {
        let game_state = CountingGameState::initial();
        let game_engine = CountingGameEngine::new();
        let analyzer = CountingAnalyzer::new();
        let cpuct = CPUCTTest { cpuct: 3.0 };
        let temp = TempTest;

        let mut mcts = MCTS::new(
            game_state,
            &game_engine,
            &analyzer,
            MCTSOptions::new(None, 0.0, 0.0, 0.0, 1.0, 10.0, 0.05, 1),
            cpuct,
            temp,
        );

        mcts.search_visits(2).await.unwrap();

        let metrics = mcts.get_root_node_metrics().unwrap();

        assert_metrics(
            &metrics,
            &NodeMetrics {
                visits: 2,
                value: Value([0.0, 0.0]),
                moves_left: 0.0,
                children: vec![
                    EdgeMetrics::new(CountingAction::Stay, 0.5, 0.0, 1),
                    EdgeMetrics::new(CountingAction::Increment, 0.0, 0.0, 0),
                    EdgeMetrics::new(CountingAction::Decrement, 0.0, 0.0, 0),
                ],
            },
        );
    }

    #[tokio::test]
    async fn test_mcts_clear_nodes_results_in_same_outcome() {
        let game_state = CountingGameState::initial();
        let game_engine = CountingGameEngine::new();
        let analyzer = CountingAnalyzer::new();
        let search_num_visits = 800;
        let cpuct = || CPUCTTest { cpuct: 3.0 };
        let temp = || TempTest;

        let mut non_clear_mcts = MCTS::new(
            game_state.clone(),
            &game_engine,
            &analyzer,
            MCTSOptions::new(None, 0.0, 0.0, 0.0, 1.0, 10.0, 0.05, 1),
            cpuct(),
            temp(),
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
            MCTSOptions::new(None, 0.0, 0.0, 0.0, 1.0, 10.0, 0.05, 1),
            cpuct(),
            temp(),
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
            MCTSOptions::new(None, 0.0, 0.0, 0.0, 1.0, 10.0, 0.05, 1),
            cpuct(),
            temp(),
        );

        initial_mcts.advance_to_action(action).await.unwrap();
        initial_mcts.search_visits(search_num_visits).await.unwrap();

        let initial_metrics = initial_mcts.get_root_node_metrics().unwrap();

        assert_metrics(&initial_metrics, &clear_metrics);
        assert_metrics(&non_clear_metrics, &clear_metrics);
    }
}
