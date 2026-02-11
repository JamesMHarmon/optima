use model::ActionWithPolicy;
use std::sync::atomic::{AtomicU32, Ordering};

use super::{
    AfterState, EdgeInfo, NodeArena, NodeId, NodeType, PUCTEdge, RollupStats, Terminal,
    edge_store::EdgeStore,
};

pub struct StateNode<A, R, SI> {
    transposition_hash: u64,
    visits: AtomicU32,
    rollup_stats: R,
    state_info: SI,
    edges: EdgeStore<A>,
}

pub type EdgeRef<'a> = &'a PUCTEdge;

impl<A, R, SI> StateNode<A, R, SI> {
    pub fn new(
        transposition_hash: u64,
        policy_priors: impl Into<Box<[ActionWithPolicy<A>]>>,
        state_info: SI,
        rollup_stats: R,
    ) -> Self {
        let edges = EdgeStore::new(policy_priors.into());
        let visits = AtomicU32::new(1);

        Self {
            transposition_hash,
            visits,
            rollup_stats,
            state_info,
            edges,
        }
    }

    /// Ensures there is at most one frontier edge (defined as `visits == 0`), and that if there
    /// is no frontier edge (i.e. the last edge has `visits > 0`), a new one is materialized with
    /// the highest policy prior among not-yet-materialized actions.
    ///
    /// This intentionally avoids sorting (which is often wasted work if only a few edges are ever
    /// traversed) at the cost of a scan when the frontier advances.
    pub fn ensure_frontier_edge(&self) {
        self.edges.ensure_frontier_edge();
    }

    pub fn edge_count(&self) -> usize {
        self.edges.edge_count()
    }

    pub fn edge_and_action(&self, index: usize) -> (&PUCTEdge, &A) {
        let edge = self.edges.get_edge(index);
        let action_idx = edge.action_idx();
        let action = &self.edges.get_action(action_idx).action();
        (edge, action)
    }

    pub fn iter_edge_refs(&self) -> impl DoubleEndedIterator<Item = &PUCTEdge> + ExactSizeIterator {
        self.edges.iter_edge_refs()
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

    pub fn rollup_stats(&self) -> &R {
        &self.rollup_stats
    }
}

impl<A, R, SI> StateNode<A, R, SI>
where
    R: RollupStats,
{
    pub fn iter_edges<'a>(
        &'a self,
        nodes: &'a NodeArena<StateNode<A, R, SI>, AfterState, Terminal<R>>,
    ) -> impl Iterator<Item = EdgeInfo<'a, A, R::Snapshot>> + 'a {
        let policy_priors: &'a [ActionWithPolicy<A>] = self.edges.policy_priors();

        self.edges
            .edges_iter()
            .enumerate()
            .map(move |(edge_index, edge)| {
                let action_idx = edge.action_idx() as usize;
                let action_with_policy = &policy_priors[action_idx];
                let visits = edge.visits();
                let snapshot = StateNode::child_snapshot(edge.child(), nodes);

                EdgeInfo {
                    edge_index,
                    action: &action_with_policy.action(),
                    policy_prior: action_with_policy.policy_score().to_f32(),
                    visits,
                    snapshot,
                }
            })
    }

    fn child_snapshot(
        child: Option<NodeId>,
        nodes: &NodeArena<StateNode<A, R, SI>, AfterState, Terminal<R>>,
    ) -> Option<R::Snapshot> {
        child.map(|child_id| match child_id.node_type() {
            NodeType::State => nodes.get_state_node(child_id).rollup_stats().snapshot(),

            NodeType::AfterState => {
                let after_state = nodes.get_after_state_node(child_id);
                let weighted_outcomes = after_state
                    .iter_outcomes(nodes)
                    .map(|(r, w)| (r.snapshot(), w));

                <R as RollupStats>::aggregate_weighted(weighted_outcomes)
            }

            NodeType::Terminal => nodes.get_terminal_node(child_id).rollup_stats().snapshot(),
        })
    }
}
