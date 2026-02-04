use model::ActionWithPolicy;
use std::sync::atomic::AtomicU32;
use tinyvec::TinyVec;

use super::{NodeId, PUCTEdge};

pub struct StateNode<A, R> {
    transposition_hash: u64,
    visits: AtomicU32,
    rollup_stats: R,
    policy_priors: Vec<ActionWithPolicy<A>>,
    edges: Vec<PUCTEdge>,
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

    fn expand_edge(&mut self, index: usize) -> usize {
        if index < self.edges.len() {
            return index;
        }

        let expansion_index = self.edges.len();
        self.policy_priors.swap(index, expansion_index);

        self.edges.push(PUCTEdge::new());

        expansion_index
    }

    fn get_edge(&self, index: usize) -> Option<&PUCTEdge> {
        self.edges.get(index)
    }

    fn get_action(&self, index: usize) -> &ActionWithPolicy<A> {
        &self.policy_priors[index]
    }
}

/// Stochastic node representing (State, Action) pair before environment response.
/// Contains multiple possible next states with their probabilities.
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
pub struct Terminal<T> {
    pub value: T,
}
