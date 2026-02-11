use append_only_vec::AppendOnlyVec;
use model::ActionWithPolicy;
use std::sync::atomic::{AtomicU32, Ordering};

use super::{AfterState, EdgeInfo, NodeArena, NodeId, NodeType, PUCTEdge, RollupStats, Terminal};

pub struct StateNode<A, R, SI> {
    transposition_hash: u64,
    visits: AtomicU32,
    rollup_stats: R,
    state_info: SI,
    policy_priors: Box<[ActionWithPolicy<A>]>,
    edges: AppendOnlyVec<PUCTEdge>,
}

pub type EdgeRef<'a> = &'a PUCTEdge;

impl<A, R, SI> StateNode<A, R, SI> {
    pub fn new(
        transposition_hash: u64,
        policy_priors: impl Into<Box<[ActionWithPolicy<A>]>>,
        state_info: SI,
        rollup_stats: R,
    ) -> Self {
        let policy_priors = policy_priors.into();

        Self {
            transposition_hash,
            visits: AtomicU32::new(1),
            rollup_stats,
            state_info,
            policy_priors,
            edges: AppendOnlyVec::new(),
        }
    }

    pub fn get_edge(&self, index: usize) -> EdgeRef<'_> {
        &self.edges[index]
    }

    pub fn get_edge_and_action(&self, index: usize) -> (EdgeRef<'_>, &A) {
        let edge = self.get_edge(index);
        let action_idx = edge.action_idx();
        let action = &self.get_action(action_idx).action;
        (edge, action)
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
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

    fn get_action(&self, action_idx: u32) -> &ActionWithPolicy<A> {
        &self.policy_priors[action_idx as usize]
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
        let policy_priors: &'a [ActionWithPolicy<A>] = &self.policy_priors;

        self.edges.iter().map(move |edge| {
            let action_idx = edge.action_idx() as usize;
            let action_with_policy = &policy_priors[action_idx];
            let visits = edge.visits();
            let snapshot = StateNode::child_snapshot(edge.child(), nodes);

            EdgeInfo {
                action: &action_with_policy.action,
                policy_prior: action_with_policy.policy_score.to_f32(),
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
