use append_only_vec::AppendOnlyVec;
use model::ActionWithPolicy;
use std::sync::atomic::{AtomicU32, Ordering};
use tinyvec::TinyVec;

use super::{EdgeInfo, NodeArena, NodeId, NodeType, PUCTEdge};

pub struct StateNode<A, R, SI> {
    pub transposition_hash: u64,
    pub visits: AtomicU32,
    pub rollup_stats: R,
    pub state_info: SI,
    pub policy_priors: Vec<ActionWithPolicy<A>>,
    pub edges: Vec<PUCTEdge>,
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
        let edge = self.get_edge(index).expect("Selected edge must be expanded");
        let action = &self.get_action(index).action;
        (edge, action)
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    pub fn increment_visits(&self) {
        self.visits.fetch_add(1, Ordering::AcqRel);
    }

    pub fn iter_edges<'a>(
        &'a self,
        nodes: &'a NodeArena<StateNode<A, R, SI>, AfterState, Terminal<R>>,
    ) -> impl Iterator<Item = EdgeInfo<'a, A, R>> + 'a {
        (0..self.edges.len()).map(move |i| self.edge_info(i, nodes))
    }

    fn edge_info<'a>(
        &'a self,
        edge_idx: usize,
        nodes: &'a NodeArena<StateNode<A, R, SI>, AfterState, Terminal<R>>,
    ) -> EdgeInfo<'a, A, R> {
        let edge = &self.edges[edge_idx];
        let action_with_policy = &self.policy_priors[edge_idx];
        let child_raw = edge.child.load(Ordering::Acquire);

        let rollup_stats = if child_raw == u32::MAX {
            None
        } else {
            let child_id = NodeId::from_u32(child_raw);
            Some(match child_id.node_type() {
                NodeType::State => &nodes.get_state_node(child_id).rollup_stats,
                NodeType::AfterState => {
                    // AfterState stats are aggregated from outcomes, not stored directly
                    // TODO: implement aggregate_after_state_stats to compute weighted average
                    panic!("AfterState rollup_stats aggregation not yet implemented")
                }
                NodeType::Terminal => &nodes.get_terminal_node(child_id).rollup_stats,
            })
        };

        EdgeInfo {
            action: &action_with_policy.action,
            policy_prior: action_with_policy.policy_score.to_f32(),
            visits: edge.visits.load(Ordering::Acquire),
            rollup_stats,
        }
    }

    /// Iterate over children with their rollup stats and visits for backpropagation.
    ///
    /// This provides the data needed for weighted-average aggregation during backpropagation.
    /// Returns an iterator of (rollup_stats, visits) pairs for each child edge.
    pub fn iter_children_stats<'a>(
        &'a self,
        nodes: &'a NodeArena<StateNode<A, R, SI>, AfterState, Terminal<R>>,
    ) -> impl Iterator<Item = (&'a R, u32)> + 'a {
        self.edges.iter().filter_map(move |edge| {
            let child_raw = edge.child.load(Ordering::Acquire);
            if child_raw == u32::MAX {
                return None;
            }

            let child_id = NodeId::from_u32(child_raw);
            let visits = edge.visits.load(Ordering::Acquire);

            // Get rollup stats based on child type
            let rollup_stats = match child_id.node_type() {
                NodeType::State => Some(&nodes.get_state_node(child_id).rollup_stats),
                NodeType::AfterState => {
                    // AfterState stats must be aggregated from outcomes
                    // This is expensive but necessary for weighted averaging
                    None // TODO: compute on-demand or cache
                }
                NodeType::Terminal => Some(&nodes.get_terminal_node(child_id).rollup_stats),
            };

            rollup_stats.map(|stats| (stats, visits))
        })
    }
}

fn aggregate_after_state_stats<'a, A, R, SI, B>(
    after_state: &'a AfterState,
    nodes: &'a NodeArena<StateNode<A, R, SI>, AfterState, Terminal<R>>,
    backprop_strategy: &B,
    target: &R,
) where
    B: crate::BackpropagationStrategy<RollupStats = R>,
{
    backprop_strategy.aggregate_stats(target, after_state.iter_outcome_stats(nodes))
}

/// Stochastic node representing (State, Action) pair before environment response.
/// Contains multiple possible next states with their probabilities.
/// Visits and rollup_stats are computed by aggregating from the outcomes.
pub struct AfterState {
    pub outcomes: TinyVec<[AfterStateOutcome; 2]>,
}

impl AfterState {
    pub fn new(outcomes: TinyVec<[AfterStateOutcome; 2]>) -> Self {
        Self { outcomes }
    }

    /// Iterate over outcome children with their rollup stats and visits.
    ///
    /// This provides the data needed for aggregating AfterState statistics from outcomes.
    /// Returns an iterator of (rollup_stats, visits) pairs for each outcome.
    pub fn iter_outcome_stats<'a, A, R, SI>(
        &'a self,
        nodes: &'a NodeArena<StateNode<A, R, SI>, AfterState, Terminal<R>>,
    ) -> impl Iterator<Item = (&'a R, u32)> + 'a {
        self.outcomes.iter().map(move |outcome| {
            let child_id = outcome.child;
            let visits = outcome.visits.load(Ordering::Acquire);

            debug_assert!(
                child_id.as_u32() != u32::MAX,
                "AfterState outcome has unset child NodeId"
            );

            // AfterState outcomes always point to either State or Terminal nodes
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
}

/// Possible outcome from an AfterState node.
pub struct AfterStateOutcome {
    pub visits: AtomicU32,
    pub child: NodeId,
}

impl AfterStateOutcome {
    pub fn new() -> Self {
        Self {
            visits: AtomicU32::new(0),
            child: u32::MAX.into(),
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
