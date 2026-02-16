use anyhow::Result;
use common::TranspositionHash;
use engine::GameEngine;
use model::GameAnalyzer;
use puct::{PUCT, RollupStats, SelectionPolicy, ValueModel};

type SnapshotOf<VM> = <<VM as ValueModel>::Rollup as RollupStats>::Snapshot;

/// Transitional wrapper that runs search via the `puct` crate while providing an
/// MCTS-like container API.
///
/// This is intentionally minimal for now; parity features (focus actions, noise,
/// detailed node/edge introspection) will be layered on next.
pub struct PuctMCTS<'a, E, M, VM, Sel>
where
    E: GameEngine,
    M: GameAnalyzer<State = E::State, Action = E::Action>,
    VM: ValueModel<State = E::State, Predictions = M::Predictions, Terminal = E::Terminal>,
    Sel: SelectionPolicy<SnapshotOf<VM>, State = E::State>,
    E::State: TranspositionHash,
    E::Terminal: engine::Value,
{
    engine: &'a E,
    state: E::State,
    puct: PUCT<'a, E, M, VM, Sel>,
}

impl<'a, E, M, VM, Sel> PuctMCTS<'a, E, M, VM, Sel>
where
    E: GameEngine,
    M: GameAnalyzer<State = E::State, Action = E::Action>,
    VM: ValueModel<State = E::State, Predictions = M::Predictions, Terminal = E::Terminal>,
    Sel: SelectionPolicy<SnapshotOf<VM>, State = E::State>,
    E::State: TranspositionHash,
    E::Terminal: engine::Value,
{
    pub fn new(
        state: E::State,
        engine: &'a E,
        analyzer: &'a M,
        value_model: &'a VM,
        selection: &'a Sel,
    ) -> Self {
        let puct = PUCT::new(engine, analyzer, value_model, selection);

        Self {
            engine,
            state,
            puct,
        }
    }

    pub fn state(&self) -> &E::State {
        &self.state
    }

    /// Runs exactly `simulations` PUCT iterations from the current root.
    pub fn search_simulations(&mut self, simulations: usize) {
        for _ in 0..simulations {
            self.puct.search(&self.state);
        }
    }

    pub fn principal_variation(&self, max_depth: usize) -> Vec<E::Action>
    where
        E::Action: Clone,
        E::State: Clone,
        SnapshotOf<VM>: Clone,
    {
        let mut pv = Vec::new();
        let mut state = self.state.clone();

        for _ in 0..max_depth {
            let edges = self.puct.edge_views(&state);
            let Some(best) = edges
                .into_iter()
                .filter(|e| e.visits > 0)
                .max_by_key(|e| e.visits)
            else {
                break;
            };

            let action = best.action;
            pv.push(action.clone());
            state = self.engine.take_action(&state, &action);
        }

        pv
    }

    pub fn advance_to_action_retain(&mut self, action: E::Action) -> Result<()> {
        self.state = self.engine.take_action(&self.state, &action);
        self.puct.prune(&self.state);
        Ok(())
    }
}
