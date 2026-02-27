use anyhow::{Result, anyhow};
use common::{PlayerToMove, TranspositionHash};
use engine::GameEngine;
use model::GameAnalyzer;
use rand::Rng;
use rand::distributions::WeightedIndex;
use rand::prelude::Distribution;
use rand::thread_rng;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use crate::node_details::{EdgeDetails, NodeDetails};
use crate::options::DirichletOptions;
use crate::temp::Temperature;
use crate::{EdgeView, NodeInfo, PUCT, SelectionPolicy, ValueModel, WeightedMerge};
use model::{EdgeMetrics, NodeMetrics};

type SnapshotOf<VM> = <VM as ValueModel>::Snapshot;
type ActionOf<E> = <E as GameEngine>::Action;
type PredictionsOf<M> = <M as GameAnalyzer>::Predictions;
type RootNodeMetrics<E, M, VM> = NodeMetrics<ActionOf<E>, PredictionsOf<M>, SnapshotOf<VM>>;

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
{
    engine: &'a E,
    analyzer: &'a M,
    state: E::State,
    puct: PUCT<'a, E, M, VM, Sel>,
    focus_actions: Vec<E::Action>,
}

impl<'a, E, M, VM, Sel> PuctMCTS<'a, E, M, VM, Sel>
where
    E: GameEngine + Sync,
    M: GameAnalyzer<State = E::State, Action = E::Action> + Sync,
    VM: ValueModel<State = E::State, Predictions = M::Predictions, Terminal = E::Terminal> + Sync,
    <VM as ValueModel>::Rollup: Send + Sync,
    Sel: SelectionPolicy<SnapshotOf<VM>, State = E::State> + Sync,
    E::State: TranspositionHash + Clone + Send + Sync,
    E::Action: Clone + PartialEq + Send + Sync,
    SnapshotOf<VM>: Clone + Send + Sync,
{
    pub fn new(
        state: E::State,
        engine: &'a E,
        analyzer: &'a M,
        value_model: &'a VM,
        selection: &'a Sel,
        parallelism: usize,
    ) -> Self {
        let puct = PUCT::new(engine, analyzer, value_model, selection, parallelism);

        Self {
            engine,
            analyzer,
            state,
            puct,
            focus_actions: Vec::new(),
        }
    }

    pub fn state(&self) -> &E::State {
        &self.state
    }

    /// PUCT implementation currently does not support persistent root prior mutation.
    /// This is a compatibility stub for legacy callers; it intentionally no-ops.
    pub async fn apply_noise_at_root(&mut self, _dirichlet: Option<&DirichletOptions>) {
        // TODO: Implement true root prior noise once puct exposes prior mutation.
    }

    fn focus_state(&self) -> E::State
    where
        E::State: Clone,
    {
        let mut state = self.state.clone();
        for action in &self.focus_actions {
            state = self.engine.take_action(&state, action);
        }
        state
    }

    pub fn add_focus_to_action(&mut self, action: E::Action) {
        self.focus_actions.push(action);
    }

    pub fn clear_focus(&mut self) {
        self.focus_actions.clear();
    }

    pub fn get_focused_actions(&self) -> &[E::Action] {
        &self.focus_actions
    }

    /// Returns an owned snapshot of focused root edge stats.
    pub fn edge_views(&self) -> Vec<EdgeView<E::Action, SnapshotOf<VM>>> {
        let state = self.focus_state();
        self.puct.edge_views(&state)
    }

    pub fn num_focus_node_visits(&self) -> usize {
        let edges = self.edge_views();
        let sum: u64 = edges.iter().map(|e| e.visits as u64).sum();
        (sum.saturating_add(1)).min(usize::MAX as u64) as usize
    }

    pub fn principal_variation(&self, max_depth: usize) -> Vec<E::Action> {
        let mut pv = Vec::new();
        let mut state = self.focus_state();

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

    pub async fn advance_to_action_retain(&mut self, action: E::Action) -> Result<()> {
        self.state = self.engine.take_action(&self.state, &action);
        self.focus_actions.clear();
        Ok(())
    }

    /// Advances to `action` and prunes the underlying search store to the new root.
    pub async fn advance_to_action(&mut self, action: E::Action) -> Result<()> {
        self.state = self.engine.take_action(&self.state, &action);
        self.focus_actions.clear();

        tokio::task::block_in_place(move || {
            self.puct.prune(&self.state);
        });

        Ok(())
    }

    pub async fn search_visits(&mut self, visits: usize) -> Result<usize> {
        self.search(|node_info| node_info.visits < visits as u32)
            .await
    }

    pub async fn search_visits_active(
        &mut self,
        visits: usize,
        active: &AtomicBool,
    ) -> Result<usize> {
        self.search(|node_info| active.load(Ordering::Acquire) && node_info.visits < visits as u32)
            .await
    }

    pub async fn search_time_max_visits(
        &mut self,
        duration: Duration,
        max_visits: usize,
        active: &AtomicBool,
    ) -> Result<usize> {
        let start = Instant::now();
        self.search(|node_info| {
            active.load(Ordering::Acquire)
                && start.elapsed() < duration
                && (max_visits == 0 || node_info.visits < max_visits as u32)
        })
        .await
    }

    pub async fn search<Fn>(&mut self, alive: Fn) -> Result<usize>
    where
        Fn: FnMut(NodeInfo) -> bool + Send,
    {
        let state = self.focus_state();

        let depth = tokio::task::block_in_place(move || self.puct.search(&state, alive));

        Ok(depth as usize)
    }

    pub fn select_action<T>(&mut self, temp: &T) -> Result<E::Action>
    where
        T: Temperature<State = E::State>,
        E::Action: Clone,
        E::State: Clone,
        SnapshotOf<VM>: Clone,
    {
        let state = self.focus_state();
        let edges = self.puct.edge_views(&state);

        if edges.is_empty() {
            return Err(anyhow!(
                "Root or focused node does not exist. Run search first."
            ));
        }

        let temp_and_offset = temp.temp(&state);

        if temp_and_offset.temperature > 0.0 {
            let weights = edges.iter().map(|e| {
                (e.visits as f32 + temp_and_offset.temperature_visit_offset)
                    .max(0.0)
                    .powf(1.0 / temp_and_offset.temperature)
            });

            let dist = WeightedIndex::new(weights);
            let chosen_idx = match dist {
                Err(_) => thread_rng().gen_range(0..edges.len()),
                Ok(dist) => dist.sample(&mut thread_rng()),
            };

            return Ok(edges[chosen_idx].action.clone());
        }

        // Temperature == 0: choose the most visited edge.
        edges
            .into_iter()
            .max_by_key(|e| e.visits)
            .map(|e| e.action)
            .ok_or_else(|| anyhow!("No available actions"))
    }

    pub fn get_root_node_metrics(&mut self) -> Result<RootNodeMetrics<E, M, VM>>
    where
        E::Action: Clone,
        E::State: Clone + PlayerToMove,
        M::Predictions: Clone,
        SnapshotOf<VM>: Clone,
    {
        let state = self.focus_state();
        let transposition_hash = state.transposition_hash();
        let request_id = transposition_hash;

        self.analyzer.analyze(request_id, &state);
        let (recv_request_id, analysis) = self.analyzer.recv();
        assert!(
            recv_request_id == request_id,
            "Expected analysis result for root node"
        );

        let predictions = analysis.predictions().clone();

        let edges: Vec<EdgeView<E::Action, SnapshotOf<VM>>> = self.puct.edge_views(&state);

        let children = edges
            .into_iter()
            .map(|e| {
                let snap = e
                    .snapshot
                    .as_ref()
                    .cloned()
                    .unwrap_or_else(SnapshotOf::<VM>::zero);
                EdgeMetrics::new(e.action, e.visits as usize, snap)
            })
            .collect::<Vec<_>>();

        let visits_sum: u64 = children.iter().map(|c| c.visits() as u64).sum();
        let visits = (visits_sum.saturating_add(1)).min(usize::MAX as u64) as usize;

        Ok(NodeMetrics {
            visits,
            predictions,
            children,
        })
    }

    pub fn get_focus_node_details(
        &mut self,
    ) -> Result<Option<NodeDetails<E::Action, SnapshotOf<VM>>>> {
        let state = self.focus_state();
        let edges = self.puct.edge_views(&state);
        if edges.is_empty() {
            return Ok(None);
        }

        let player_to_move = self.engine.player_to_move(&state);
        let edge_details = edges
            .into_iter()
            .map(|e| {
                let snapshot: SnapshotOf<VM> = e.snapshot.unwrap_or_else(SnapshotOf::<VM>::zero);

                //@TODO: Analyze this implementation of values.

                EdgeDetails {
                    action: e.action,
                    Nsa: e.visits as usize,
                    Psa: e.policy_prior,
                    Usa: 0.0,
                    cpuct: 0.0,
                    puct_score: 0.0,
                    snapshot,
                    player_to_move,
                }
            })
            .collect::<Vec<_>>();

        let visits_sum: u64 = edge_details.iter().map(|d| d.Nsa as u64).sum();
        let visits = (visits_sum.saturating_add(1)).min(usize::MAX as u64) as usize;

        Ok(Some(NodeDetails {
            visits,
            children: edge_details,
        }))
    }

    pub fn get_principal_variation(
        &mut self,
        action: Option<&E::Action>,
        depth: usize,
    ) -> Result<Vec<EdgeDetails<E::Action, SnapshotOf<VM>>>> {
        let mut state = self.focus_state();
        let mut pv: Vec<EdgeDetails<E::Action, SnapshotOf<VM>>> = Vec::new();

        for ply in 0..depth {
            let edges = self.puct.edge_views(&state);
            if edges.is_empty() {
                break;
            }

            let chosen = if ply == 0 {
                if let Some(desired) = action {
                    edges.into_iter().find(|e| &e.action == desired)
                } else {
                    edges
                        .into_iter()
                        .filter(|e| e.visits > 0)
                        .max_by_key(|e| e.visits)
                }
            } else {
                edges
                    .into_iter()
                    .filter(|e| e.visits > 0)
                    .max_by_key(|e| e.visits)
            };

            let Some(chosen) = chosen else {
                break;
            };

            let snapshot = chosen.snapshot.unwrap_or_else(SnapshotOf::<VM>::zero);
            let player_to_move = self.engine.player_to_move(&state);

            //@TODO: Analyze this implementation of values.
            let details = EdgeDetails {
                action: chosen.action.clone(),
                Nsa: chosen.visits as usize,
                Psa: chosen.policy_prior,
                Usa: 0.0,
                cpuct: 0.0,
                puct_score: 0.0,
                snapshot,
                player_to_move,
            };

            state = self.engine.take_action(&state, &chosen.action);
            pv.push(details);
        }

        Ok(pv)
    }
}
