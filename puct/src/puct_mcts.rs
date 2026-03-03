use anyhow::{Result, anyhow};
use common::{PlayerToMove, TranspositionHash};
use engine::{GameEngine, ValidActions};
use model::GameAnalyzer;
use rand::Rng;
use rand::distributions::WeightedIndex;
use rand::prelude::Distribution;
use rand::thread_rng;
use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use crate::node_details::{EdgeDetails, NodeDetails};
use crate::options::DirichletOptions;
use crate::temp::Temperature;
use crate::{EdgeScore, EdgeView};
use crate::{NodeInfo, PUCT, SelectionPolicy, SelectionPolicyScoring, ValueModel, WeightedMerge};
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
    VM: ValueModel<Predictions = M::Predictions, Terminal = E::Terminal>,
    Sel: SelectionPolicy<
            SnapshotOf<VM>,
            State = M::State,
            Action = E::Action,
            Terminal = E::Terminal,
        >,
    M::State: TranspositionHash,
{
    engine: &'a E,
    analyzer: &'a M,
    state: M::State,
    puct: PUCT<'a, E, M, VM, Sel>,
}

impl<'a, E, M, VM, Sel> PuctMCTS<'a, E, M, VM, Sel>
where
    E: GameEngine<State = M::State, Action = M::Action>
        + ValidActions<State = M::State, Action = M::Action>
        + Sync,
    M: GameAnalyzer + Sync,
    VM: ValueModel<Predictions = M::Predictions, Terminal = E::Terminal> + Sync,
    <VM as ValueModel>::Rollup: Send + Sync,
    Sel: SelectionPolicy<
            SnapshotOf<VM>,
            State = M::State,
            Action = M::Action,
            Terminal = E::Terminal,
        > + Sync,
    M::State: TranspositionHash + Clone + Send + Sync,
    M::Action: Clone + Eq + Hash + Send + Sync,
    SnapshotOf<VM>: Clone + Send + Sync,
{
    pub fn new(
        state: M::State,
        engine: &'a E,
        analyzer: &'a M,
        value_model: &'a VM,
        selection: &'a Sel,
        parallelism: usize,
        sim_threads: usize,
    ) -> Self {
        let puct = PUCT::new(
            engine,
            analyzer,
            value_model,
            selection,
            parallelism,
            sim_threads,
        );

        Self {
            engine,
            analyzer,
            state,
            puct,
        }
    }

    /// PUCT implementation currently does not support persistent root prior mutation.
    /// This is a compatibility stub for legacy callers; it intentionally no-ops.
    pub async fn apply_noise_at_root(&mut self, _dirichlet: Option<&DirichletOptions>) {
        // @TODO: Implement true root prior noise once puct exposes prior mutation.
        panic!("apply_noise_at_root is not currently implemented for PuctMCTS");
    }

    pub fn state(&self) -> &M::State {
        &self.state
    }

    pub fn set_state(&mut self, state: M::State) {
        self.state = state;
    }

    pub fn num_node_visits(&mut self) -> usize {
        let edges = self.puct.edge_views(&self.state);
        let sum: u64 = edges.iter().map(|e| e.visits as u64).sum();
        (sum + 1).min(usize::MAX as u64) as usize
    }

    pub async fn advance_to_action_retain(&mut self, action: M::Action) -> Result<()> {
        self.state = self.engine.take_action(&self.state, &action);
        Ok(())
    }

    /// Advances to `action` and prunes the underlying search store to the new root.
    pub async fn advance_to_action(&mut self, action: M::Action) -> Result<()> {
        self.state = self.engine.take_action(&self.state, &action);

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

    pub async fn search<F>(&mut self, alive: F) -> Result<usize>
    where
        F: Fn(NodeInfo) -> bool + Send + Sync,
    {
        let state = self.state.clone();

        let depth = tokio::task::block_in_place(move || self.puct.search(&state, alive));

        Ok(depth as usize)
    }

    pub fn select_action<T>(&mut self, temp: &T) -> Result<M::Action>
    where
        T: Temperature<State = M::State>,
        M::Action: Clone,
        M::State: Clone,
        SnapshotOf<VM>: Clone,
    {
        let state = self.state.clone();
        let edges = self.filtered_edge_views(&state);

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
        M::Action: Clone,
        M::State: Clone + PlayerToMove,
        M::Predictions: Clone,
        SnapshotOf<VM>: Clone,
    {
        let state = self.state.clone();
        let analysis = self.analyzer.analyze(&state);

        let predictions = analysis.predictions().clone();

        let edges: Vec<EdgeView<M::Action, SnapshotOf<VM>>> = self.puct.edge_views(&state);

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

    pub fn get_node_details(&mut self) -> NodeDetails<M::Action, SnapshotOf<VM>>
    where
        Sel: SelectionPolicyScoring<SnapshotOf<VM>, State = M::State>,
    {
        let state = self.state.clone();
        let edges = self.filtered_edge_views(&state);

        let mut score_by_index = self.score_by_index(&state, 0);

        let player_to_move = self.engine.player_to_move(&state);
        let edge_details = edges
            .into_iter()
            .map(|e| {
                let snapshot: SnapshotOf<VM> = e.snapshot.unwrap_or_else(SnapshotOf::<VM>::zero);

                let score = score_by_index.remove(&e.edge_index).unwrap_or_default();

                EdgeDetails {
                    action: e.action,
                    Nsa: e.visits as usize,
                    Psa: e.policy_prior,
                    Usa: score.usa,
                    cpuct: score.cpuct,
                    puct_score: score.puct_score,
                    snapshot,
                    player_to_move,
                }
            })
            .collect::<Vec<_>>();

        let visits_sum: u64 = edge_details.iter().map(|d| d.Nsa as u64).sum();
        let visits = (visits_sum.saturating_add(1)).min(usize::MAX as u64) as usize;

        NodeDetails {
            visits,
            children: edge_details,
        }
    }

    pub fn principal_variation(
        &mut self,
        action: Option<&M::Action>,
        depth: usize,
    ) -> Result<Vec<EdgeDetails<M::Action, SnapshotOf<VM>>>>
    where
        Sel: SelectionPolicyScoring<SnapshotOf<VM>, State = M::State>,
    {
        let mut state = self.state.clone();
        let mut pv: Vec<EdgeDetails<M::Action, SnapshotOf<VM>>> = Vec::new();

        for ply in 0..depth {
            let edges = match self.try_filtered_edge_views(&state) {
                Some(edges) => edges,
                None => break,
            };

            let mut score_by_index = self.score_by_index(&state, ply as u32);

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

            let score = score_by_index
                .remove(&chosen.edge_index)
                .unwrap_or_default();

            let details = EdgeDetails {
                action: chosen.action.clone(),
                Nsa: chosen.visits as usize,
                Psa: chosen.policy_prior,
                Usa: score.usa,
                cpuct: score.cpuct,
                puct_score: score.puct_score,
                snapshot,
                player_to_move,
            };

            state = self.engine.take_action(&state, &chosen.action);
            pv.push(details);
        }

        Ok(pv)
    }

    /// Returns edges for `state` filtered to only those whose action is currently valid.
    ///
    /// Because the transposition table can map a node to multiple game states
    /// (some of which may have fewer legal actions due to repetition rules), the
    /// raw edge list can contain actions that are illegal in the current context.
    /// This helper ensures callers always work with the correct legal subset.
    fn filtered_edge_views(
        &mut self,
        state: &M::State,
    ) -> Vec<EdgeView<M::Action, SnapshotOf<VM>>> {
        let valid_actions: HashSet<M::Action> = self.engine.valid_actions(state).collect();
        self.puct
            .edge_views(state)
            .into_iter()
            .filter(|e| valid_actions.contains(&e.action))
            .collect()
    }

    fn try_filtered_edge_views(
        &mut self,
        state: &M::State,
    ) -> Option<Vec<EdgeView<M::Action, SnapshotOf<VM>>>> {
        let valid_actions: HashSet<M::Action> = self.engine.valid_actions(state).collect();
        self.puct.try_edge_views(state).map(|edges| {
            edges
                .into_iter()
                .filter(|e| valid_actions.contains(&e.action))
                .collect()
        })
    }

    fn score_by_index(&self, state: &M::State, depth: u32) -> HashMap<usize, EdgeScore>
    where
        Sel: SelectionPolicyScoring<SnapshotOf<VM>, State = M::State>,
    {
        self.puct
            .edge_scores(state, depth)
            .into_iter()
            .map(|s| (s.edge_index, s))
            .collect()
    }
}
