use crate::DynamicCPUCT;
use crate::counting_game::{CountingAnalyzer, CountingGameEngine, CountingGameState};

use engine::GameState;

use puct::{MovesLeftSelectionPolicy, MovesLeftStrategyOptions, MovesLeftValueModel};

#[test]
fn puct_mcts_can_search_and_return_pv() {
    let engine = CountingGameEngine::new();
    let analyzer = CountingAnalyzer::new([0.34, 0.33, 0.33]);

    let value_model = MovesLeftValueModel::<
        CountingGameState,
        crate::counting_game::CountingGamePredictions,
        crate::counting_game::CountingGamePredictions,
    >::new();

    let options = MovesLeftStrategyOptions {
        fpu: 0.0,
        fpu_root: 0.0,
        moves_left_threshold: 1.0,
        moves_left_scale: 10.0,
        moves_left_factor: 0.0,
    };

    let cpuct = DynamicCPUCT::<CountingGameState>::default();

    let selection = MovesLeftSelectionPolicy::new(cpuct, options);

    let state = CountingGameState::initial();
    let mut mcts = crate::PuctMCTS::new(state, &engine, &analyzer, &value_model, &selection);

    mcts.search_simulations(50);

    let pv = mcts.principal_variation(8);
    assert!(!pv.is_empty());
}
