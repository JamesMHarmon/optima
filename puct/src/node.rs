use model::ActionWithPolicy;
use std::collections::HashSet;
use std::sync::atomic::{AtomicU32, Ordering};
use tinyvec::TinyVec;

use super::{EdgeInfo, NodeArena, NodeId, NodeType, PUCTEdge, RollupStats};

pub struct StateNode<A, R, SI> {
    transposition_hash: u64,
    visits: AtomicU32,
    rollup_stats: R,
    state_info: SI,
    policy_priors: Vec<ActionWithPolicy<A>>,
    edges: Vec<PUCTEdge>,
}

impl<A, R, SI> StateNode<A, R, SI> {
    pub fn new(
        transposition_hash: u64,
        policy_priors: Vec<ActionWithPolicy<A>>,
        state_info: SI,
        rollup_stats: R,
    ) -> Self {
        Self {
            transposition_hash,
            visits: AtomicU32::new(1),
            rollup_stats,
            state_info,
            policy_priors,
            edges: Vec::new(),
        }
    }

    pub fn expand_edge(&mut self, index: usize) -> usize {
        if index < self.edges.len() {
            return index;
        }

        let expansion_index = self.edges.len();
        self.policy_priors.swap(index, expansion_index);

        self.edges.push(PUCTEdge::new());

        expansion_index
    }

    pub fn get_edge(&self, index: usize) -> Option<&PUCTEdge> {
        self.edges.get(index)
    }

    pub fn get_action(&self, index: usize) -> &ActionWithPolicy<A> {
        &self.policy_priors[index]
    }

    pub fn get_edge_and_action(&self, index: usize) -> (&PUCTEdge, &A) {
        let edge = self
            .get_edge(index)
            .expect("Selected edge must be expanded");
        let action = &self.get_action(index).action;
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
}

impl<A, R, SI> StateNode<A, R, SI>
where
    R: RollupStats,
{
    pub fn iter_edges<'a>(
        &'a self,
        nodes: &'a NodeArena<StateNode<A, R, SI>, AfterState, Terminal<R>>,
    ) -> impl Iterator<Item = EdgeInfo<'a, A, R::Snapshot>> + 'a {
        (0..self.edges.len()).map(move |i| self.edge_info(i, nodes))
    }

    fn edge_info<'a>(
        &'a self,
        edge_idx: usize,
        nodes: &'a NodeArena<StateNode<A, R, SI>, AfterState, Terminal<R>>,
    ) -> EdgeInfo<'a, A, R::Snapshot> {
        let edge = &self.edges[edge_idx];
        let action_with_policy = &self.policy_priors[edge_idx];

        let snapshot = edge.child().map(|child_id| match child_id.node_type() {
            NodeType::State => nodes.get_state_node(child_id).rollup_stats.snapshot(),

            NodeType::AfterState => {
                let after_state = nodes.get_after_state_node(child_id);
                let weighted_outcomes = after_state
                    .iter_outcomes(nodes)
                    .map(|(r, w)| (r.snapshot(), w));

                <R as RollupStats>::aggregate_weighted(weighted_outcomes)
            }

            NodeType::Terminal => nodes.get_terminal_node(child_id).rollup_stats.snapshot(),
        });

        EdgeInfo {
            action: &action_with_policy.action,
            policy_prior: action_with_policy.policy_score.to_f32(),
            visits: edge.visits(),
            snapshot,
        }
    }
}

/// Stochastic node representing (State, Action) pair before environment response.
/// Contains multiple possible next states.
/// Aggregation is performed externally via Snapshot merging.
pub struct AfterState {
    pub outcomes: TinyVec<[AfterStateOutcome; 2]>,
}

impl AfterState {
    pub fn new(outcomes: TinyVec<[AfterStateOutcome; 2]>) -> Self {
        Self { outcomes }
    }

    pub fn is_valid(&self) -> bool {
        let outcome_count = self.outcomes.len();
        let ids: HashSet<NodeId> = self.outcomes.iter().map(|o| o.child()).collect();
        ids.len() == outcome_count
            && ids.iter().filter(|id| id.is_terminal()).count() <= 1
            && ids.iter().all(|id| !id.is_after_state())
    }

    /// Iterates over outcome rollups and weights.
    pub fn iter_outcomes<'a, A, R, SI>(
        &'a self,
        nodes: &'a NodeArena<StateNode<A, R, SI>, AfterState, Terminal<R>>,
    ) -> impl Iterator<Item = (&'a R, u32)> + 'a
    where
        R: RollupStats,
    {
        self.outcomes.iter().map(move |outcome| {
            let child_id = outcome.child();
            let visits = outcome.visits();

            debug_assert!(
                !child_id.is_unset(),
                "AfterState outcome has unset child NodeId"
            );

            let rollup_stats = match child_id.node_type() {
                NodeType::State => &nodes.get_state_node(child_id).rollup_stats,
                NodeType::Terminal => &nodes.get_terminal_node(child_id).rollup_stats,
                NodeType::AfterState => {
                    panic!("AfterState outcome cannot point to another AfterState")
                }
            };

            (rollup_stats, visits)
        })
    }

    pub fn terminal_outcome(&self) -> Option<&AfterStateOutcome> {
        self.outcomes
            .iter()
            .find(|outcome| outcome.child().is_terminal())
    }
}

/// Possible outcome from an AfterState node.
pub struct AfterStateOutcome {
    visits: AtomicU32,
    child: NodeId,
}

impl AfterStateOutcome {
    pub fn new(visits: u32, child: NodeId) -> Self {
        Self {
            visits: AtomicU32::new(visits),
            child,
        }
    }

    pub fn visits(&self) -> u32 {
        self.visits.load(Ordering::Acquire)
    }

    pub fn increment_visits(&self) {
        self.visits.fetch_add(1, Ordering::AcqRel);
    }

    pub fn child(&self) -> NodeId {
        self.child
    }

    pub fn as_tuple(&self) -> (NodeId, u32) {
        (self.child(), self.visits())
    }
}

impl Default for AfterStateOutcome {
    fn default() -> Self {
        Self {
            visits: AtomicU32::new(0),
            child: NodeId::unset(),
        }
    }
}

impl Clone for AfterStateOutcome {
    fn clone(&self) -> Self {
        Self {
            visits: AtomicU32::new(self.visits()),
            child: self.child,
        }
    }
}

/// Terminal node representing a final game state.
pub struct Terminal<R> {
    rollup_stats: R,
}

impl<R> Terminal<R> {
    pub fn new(rollup_stats: R) -> Self {
        Self { rollup_stats }
    }

    pub fn rollup_stats(&self) -> &R {
        &self.rollup_stats
    }
}
