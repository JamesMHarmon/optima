use std::{
    collections::HashSet,
    sync::atomic::{AtomicU32, Ordering},
};

use tinyvec::TinyVec;

use super::{NodeArena, NodeId, NodeType, RollupStats, StateNode, Terminal};

/// Stochastic node representing (State, Action) pair before environment response.
/// Contains multiple possible next states.
/// Aggregation is performed externally via Snapshot merging.
pub struct AfterState {
    pub outcomes: TinyVec<[AfterStateOutcome; 2]>,
}

type StateArena<A, R, SI> = NodeArena<StateNode<A, R, SI>, AfterState, Terminal<R>>;

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
        nodes: &'a StateArena<A, R, SI>,
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
                NodeType::State => nodes.get_state_node(child_id).rollup_stats(),
                NodeType::Terminal => nodes.get_terminal_node(child_id).rollup_stats(),
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
