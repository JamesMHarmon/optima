use model::ActionWithPolicy;
use std::sync::atomic::{AtomicU32, Ordering};

use super::{
    AfterState, EdgeInfo, NodeArena, NodeId, NodeType, PUCTEdge, RollupStats, Terminal,
    edge_store::EdgeStore,
};

pub struct StateNode<A, R: RollupStats, SI> {
    transposition_hash: u64,
    visits: AtomicU32,
    rollup_prior: R::Snapshot,
    rollup_stats: R,
    state_info: SI,
    edges: EdgeStore<A>,
}

pub type EdgeRef<'a> = &'a PUCTEdge;

impl<A, R, SI> StateNode<A, R, SI>
where
    R: RollupStats,
{
    pub fn new(
        transposition_hash: u64,
        policy_priors: impl Into<Box<[ActionWithPolicy<A>]>>,
        state_info: SI,
        rollup_stats: R,
    ) -> Self {
        let edges = EdgeStore::new(policy_priors.into());
        let visits = AtomicU32::new(1);
        let rollup_prior = rollup_stats.snapshot();

        Self {
            transposition_hash,
            visits,
            rollup_prior,
            rollup_stats,
            state_info,
            edges,
        }
    }

    /// Ensures there is at most one frontier edge (defined as `visits == 0`), and that if there
    /// is no frontier edge (i.e. the last edge has `visits > 0`), a new one is materialized with
    /// the highest policy prior among not-yet-materialized actions.
    pub fn ensure_frontier_edge(&self) {
        self.edges.ensure_frontier_edge();
    }

    pub fn edge_and_action(&self, index: usize) -> (&PUCTEdge, &A) {
        let (edge, action_with_policy) = self.edges.edge(index);
        (edge, action_with_policy.action())
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

    pub fn rollup_prior(&self) -> &R::Snapshot {
        &self.rollup_prior
    }

    pub fn rollup_stats(&self) -> &R {
        &self.rollup_stats
    }
}

impl<A, R, SI> StateNode<A, R, SI>
where
    R: RollupStats,
{
    pub fn iter_edge_rollups<'a>(
        &'a self,
        nodes: &'a NodeArena<StateNode<A, R, SI>, AfterState, Terminal<R>>,
    ) -> impl Iterator<Item = (&'a PUCTEdge, R::Snapshot)> + 'a {
        self.iter_edges().filter_map(move |edge| {
            edge.child()
                .map(|child_id| (edge, StateNode::child_snapshot(child_id, nodes)))
        })
    }

    pub fn iter_edge_info<'a>(
        &'a self,
        nodes: &'a NodeArena<StateNode<A, R, SI>, AfterState, Terminal<R>>,
    ) -> impl Iterator<Item = EdgeInfo<'a, A, R::Snapshot>> + 'a {
        self.edges.iter_edges_with_policy().enumerate().map(
            move |(edge_index, (edge, action_with_policy))| {
                let visits = edge.visits();
                let child = edge.child();
                let snapshot = child.map(|child_id| StateNode::child_snapshot(child_id, nodes));

                EdgeInfo {
                    edge_index,
                    action: action_with_policy.action(),
                    policy_prior: action_with_policy.policy_score().to_f32(),
                    visits,
                    snapshot,
                }
            },
        )
    }

    fn child_snapshot(
        child_id: NodeId,
        nodes: &NodeArena<StateNode<A, R, SI>, AfterState, Terminal<R>>,
    ) -> R::Snapshot {
        match child_id.node_type() {
            NodeType::State => nodes.get_state_node(child_id).rollup_stats().snapshot(),

            NodeType::AfterState => {
                let after_state = nodes.get_after_state_node(child_id);
                R::aggregate_rollups(after_state.iter_outcomes(nodes))
            }

            NodeType::Terminal => nodes.get_terminal_node(child_id).rollup_stats().snapshot(),
        }
    }
}

#[cfg(test)]
mod tests;
