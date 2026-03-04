use model::ActionWithPolicy;
use std::sync::atomic::{AtomicU32, Ordering};

use crate::after_state::AfterState;
use crate::edge::PUCTEdge;
use crate::edge_store::EdgeStore;
use crate::node_arena::{NodeArena, NodeId, NodeType};
use crate::rollup::{RollupStats, WeightedMerge};
use crate::selection_policy::EdgeInfo;
use crate::terminal_node::Terminal;

type StateArena<A, R> = NodeArena<StateNode<A, R>, AfterState, Terminal<R>>;

pub struct StateNode<A, R>
where
    R: RollupStats,
{
    transposition_hash: u64,
    visits: AtomicU32,
    virtual_visits: AtomicU32,
    rollup_prior: R::Snapshot,
    rollup_stats: R,
    edges: EdgeStore<A>,
}

impl<A, R> StateNode<A, R>
where
    R: RollupStats,
{
    pub fn new(
        transposition_hash: u64,
        policy_priors: impl Into<Box<[ActionWithPolicy<A>]>>,
        rollup_stats: R,
    ) -> Self {
        let edges = EdgeStore::new(policy_priors.into());
        let visits = AtomicU32::new(1);
        let virtual_visits = AtomicU32::new(0);
        let rollup_prior = rollup_stats.snapshot();

        Self {
            transposition_hash,
            visits,
            virtual_visits,
            rollup_prior,
            rollup_stats,
            edges,
        }
    }

    /// Ensures there is at most one frontier edge (defined as `visits == 0`), and that if there
    /// is no frontier edge (i.e. the last edge has `visits > 0`), a new one is materialized with
    /// the highest policy prior among not-yet-materialized actions.
    pub fn ensure_frontier_edge(&self) {
        self.edges.ensure_frontier_edge();
    }

    pub fn num_actions(&self) -> usize {
        self.edges.num_actions()
    }

    /// Resets the node back to its post-analysis state: visits back to 1,
    /// virtual_visits to 0, rollup_stats back to rollup_prior, and all edges
    /// rebuilt from scratch via `f` which may mutate the policy priors.
    pub fn reset_node<F>(&mut self, f: F)
    where
        F: FnOnce(&mut [ActionWithPolicy<A>]),
    {
        self.visits.store(1, Ordering::Release);
        self.virtual_visits.store(0, Ordering::Release);
        self.rollup_stats.set(self.rollup_prior.duplicate());
        self.edges.reset_edges(f);
    }

    pub fn edge_and_action(&self, index: usize) -> (&PUCTEdge, &A) {
        let (edge, action_with_policy) = self.edges.edge(index);
        (edge, action_with_policy.action())
    }

    pub fn edge(&self, index: usize) -> &PUCTEdge {
        self.edges.edge(index).0
    }

    pub fn iter_edges(&self) -> impl Iterator<Item = &PUCTEdge> {
        self.edges.iter_edges()
    }

    pub fn iter_edges_with_policy(
        &self,
    ) -> impl Iterator<Item = (&PUCTEdge, &ActionWithPolicy<A>)> {
        self.edges.iter_edges_with_policy()
    }

    pub fn visits(&self) -> u32 {
        self.visits.load(Ordering::Acquire)
    }

    pub fn transposition_hash(&self) -> u64 {
        self.transposition_hash
    }

    pub fn increment_visits(&self) {
        self.visits.fetch_add(1, Ordering::AcqRel);
    }

    pub fn increment_virtual_visits(&self) {
        self.virtual_visits.fetch_add(1, Ordering::AcqRel);
    }

    pub fn decrement_virtual_visits(&self) {
        let prev = self.virtual_visits.fetch_sub(1, Ordering::AcqRel);
        debug_assert!(prev > 0, "virtual_visits underflow");
    }

    pub fn virtual_visits(&self) -> u32 {
        self.virtual_visits.load(Ordering::Acquire)
    }

    pub fn rollup_prior(&self) -> &R::Snapshot {
        &self.rollup_prior
    }

    pub fn rollup_stats(&self) -> &R {
        &self.rollup_stats
    }

    pub fn recompute_rollup(&self, nodes: &StateArena<A, R>) {
        let mut aggregated = self.rollup_prior.duplicate();
        let edges = self.iter_edge_rollups(nodes);
        let visited_edges = edges.map(|(e, s)| (e.visits(), s)).filter(|(v, _)| *v > 0);

        for (visits, snapshot) in visited_edges {
            aggregated.merge_weighted(&snapshot, visits);
        }

        self.rollup_stats.set(aggregated);
    }
}

impl<A, R> StateNode<A, R>
where
    R: RollupStats,
{
    pub fn iter_edge_rollups<'a>(
        &'a self,
        nodes: &'a StateArena<A, R>,
    ) -> impl Iterator<Item = (&'a PUCTEdge, R::Snapshot)> + 'a {
        self.iter_edges().filter_map(move |edge| {
            edge.child()
                .map(|child_id| (edge, StateNode::child_snapshot(child_id, nodes)))
        })
    }

    pub fn iter_edge_info<'a>(
        &'a self,
        nodes: &'a StateArena<A, R>,
    ) -> impl Iterator<Item = EdgeInfo<'a, A, R::Snapshot>> + 'a {
        self.edges.iter_edges_with_policy().enumerate().map(
            move |(edge_index, (edge, action_with_policy))| {
                let child = edge.child();
                let snapshot = child.map(|child_id| StateNode::child_snapshot(child_id, nodes));

                EdgeInfo {
                    edge_index,
                    action: action_with_policy.action(),
                    policy_prior: action_with_policy.policy_score().to_f32(),
                    visits: edge.visits(),
                    virtual_visits: edge.virtual_visits(),
                    snapshot,
                }
            },
        )
    }

    fn child_snapshot(child_id: NodeId, nodes: &StateArena<A, R>) -> R::Snapshot {
        match child_id.node_type() {
            NodeType::State => nodes.state_node(child_id).rollup_stats().snapshot(),
            NodeType::AfterState => nodes.after_state_node(child_id).snapshot(nodes),
            NodeType::Terminal => nodes.terminal_node(child_id).rollup_stats().snapshot(),
        }
    }
}

#[cfg(test)]
mod tests;
