use model::ActionWithPolicy;
use std::sync::atomic::{AtomicU32, Ordering};
use tinyvec::TinyVec;

use super::{EdgeInfo, NodeArena, NodeId, NodeType};

pub struct StateNode<A, R> {
    pub transposition_hash: u64,
    pub visits: AtomicU32,
    pub rollup_stats: R,
    pub policy_priors: Vec<ActionWithPolicy<A>>,
    pub edges: Vec<Edge>,
}

impl<A, R> StateNode<A, R>
where
    R: Default,
{
    pub fn new(transposition_hash: u64, policy_priors: Vec<ActionWithPolicy<A>>) -> Self {
        Self {
            transposition_hash,
            visits: AtomicU32::new(1),
            rollup_stats: R::default(),
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

        self.edges.push(Edge::new());

        expansion_index
    }

    pub fn get_edge(&self, index: usize) -> Option<&Edge> {
        self.edges.get(index)
    }

    pub fn get_action(&self, index: usize) -> &ActionWithPolicy<A> {
        &self.policy_priors[index]
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    pub fn iter_edges<'a>(
        &'a self,
        nodes: &'a NodeArena<StateNode<A, R>, AfterState, Terminal<R>>,
    ) -> impl Iterator<Item = EdgeInfo<'a, A, R>> + 'a {
        (0..self.edges.len()).map(move |i| self.edge_info(i, nodes))
    }

    fn edge_info<'a>(
        &'a self,
        edge_idx: usize,
        nodes: &'a NodeArena<StateNode<A, R>, AfterState, Terminal<R>>,
    ) -> EdgeInfo<'a, A, R> {
        let edge = &self.edges[edge_idx];
        let action_with_policy = &self.policy_priors[edge_idx];
        let child_raw = edge.child.load(Ordering::Acquire);

        let rollup_stats = if child_raw == u32::MAX {
            None
        } else {
            let child_id = NodeId::from_u32(child_raw);
            Some(match child_id.node_type() {
                NodeType::State => &nodes.get_state(child_id).rollup_stats,
                NodeType::AfterState => {
                    // AfterState stats are aggregated from outcomes, not stored directly
                    // TODO: implement aggregate_after_state_stats to compute weighted average
                    panic!("AfterState rollup_stats aggregation not yet implemented")
                }
                NodeType::Terminal => &nodes.get_terminal(child_id).rollup_stats,
            })
        };

        EdgeInfo {
            action: &action_with_policy.action,
            policy_prior: action_with_policy.policy,
            visits: edge.visits.load(Ordering::Acquire),
            rollup_stats,
        }
    }
}

fn aggregate_after_state_stats<'a, A, R>(
    after_state: &'a AfterState,
    nodes: &'a NodeArena<StateNode<A, R>, AfterState, Terminal<R>>,
) -> Option<R>
where
    R: Clone + Default,
{
    // TODO: Implement proper weighted aggregation based on outcome probabilities
    // This should compute a weighted average of child rollup_stats based on outcome probabilities
    // For now, return None to indicate unimplemented
    None
}

/// Edge from a State node to an AfterState or Terminal node.
pub struct Edge {
    pub visits: AtomicU32,
    pub child: AtomicU32,
}

impl Edge {
    pub fn new() -> Self {
        Self {
            visits: AtomicU32::new(0),
            child: AtomicU32::new(u32::MAX),
        }
    }
}

/// Stochastic node representing (State, Action) pair before environment response.
/// Contains multiple possible next states with their probabilities.
/// Visits and rollup_stats are computed by aggregating from the outcomes.
pub struct AfterState {
    pub outcomes: TinyVec<[AfterStateOutcome; 2]>,
}

/// Possible outcome from an AfterState node.
pub struct AfterStateOutcome {
    pub visits: AtomicU32,
    pub child: AtomicU32,
}

impl AfterStateOutcome {
    pub fn new() -> Self {
        Self {
            visits: AtomicU32::new(0),
            child: AtomicU32::new(u32::MAX),
        }
    }
}

impl Default for AfterStateOutcome {
    fn default() -> Self {
        Self::new()
    }
}

/// Terminal node representing a final game state.
pub struct Terminal<R> {
    pub rollup_stats: R,
}
