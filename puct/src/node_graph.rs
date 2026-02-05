use std::{collections::HashSet, sync::atomic::Ordering};
use std::sync::atomic::AtomicU32;

use super::{AfterState, AfterStateOutcome, NodeArena, NodeId, NodeType, PUCTEdge, StateNode, Terminal};

/// Graph operations wrapper around NodeArena for node traversal and mutation.
pub struct NodeGraph<'a, A, R, SI> {
    arena: &'a NodeArena<StateNode<A, R, SI>, AfterState, Terminal<R>>,
}

impl<'a, A, R, SI> NodeGraph<'a, A, R, SI> {
    pub fn new(arena: &'a NodeArena<StateNode<A, R, SI>, AfterState, Terminal<R>>) -> Self {
        Self { arena }
    }

    /// Find a State node with matching transposition hash, traversing through AfterState nodes if needed.
    pub fn find_state_with_hash(&self, node_id: NodeId, transposition_hash: u64) -> Option<NodeId> {
        match node_id.node_type() {
            NodeType::State => {
                let state = self.arena.get_state(node_id);
                (state.transposition_hash == transposition_hash).then_some(node_id)
            }
            NodeType::AfterState => {
                let after_state = self.arena.get_after_state(node_id);
                after_state.outcomes.iter().find_map(|outcome| {
                    self.find_state_with_hash(outcome.child, transposition_hash)
                })
            }
            NodeType::Terminal => None,
        }
    }

    /// Check if an edge already points to a state with the given transposition hash.
    pub fn get_edge_state_with_hash(&self, edge: &PUCTEdge, transposition_hash: u64) -> Option<NodeId> {
        edge.get_child()
            .and_then(|child_id| self.find_state_with_hash(child_id, transposition_hash))
    }

    /// Add a child to an edge, converting to AfterState if multiple outcomes exist.
    pub fn add_child_to_edge(&self, edge: &PUCTEdge, child_id: NodeId) {
        if edge.try_set_child(child_id) {
            return;
        }

        let existing_child_id = edge.get_child().expect("Child must be set if try_set_child failed");

        // Build new outcomes based on existing child type
        let mut new_outcomes = tinyvec::TinyVec::new();
        
        match existing_child_id.node_type() {
            NodeType::AfterState => {
                // Copy existing outcomes (snapshot atomic visits)
                let after_state = self.arena.get_after_state(existing_child_id);
                for outcome in &after_state.outcomes {
                    new_outcomes.push(AfterStateOutcome {
                        visits: AtomicU32::new(outcome.visits.load(Ordering::Acquire)),
                        child: outcome.child,
                    });
                }
            }
            NodeType::State | NodeType::Terminal => {
                // First outcome is the existing child
                new_outcomes.push(AfterStateOutcome {
                    visits: AtomicU32::new(edge.visits.load(Ordering::Acquire)),
                    child: existing_child_id,
                });
            }
        }

        // Add new child as outcome
        new_outcomes.push(AfterStateOutcome {
            visits: AtomicU32::new(0),
            child: child_id,
        });
        
        // Create new AfterState and atomically update edge
        let new_after_state_id = self.arena.push_after_state(AfterState::new(new_outcomes));
        edge.set_child(new_after_state_id);

        debug_assert!(
            {
                let after_state = self.arena.get_after_state(new_after_state_id);
                let outcome_count = after_state.outcomes.len();
                let ids: HashSet<_> = after_state.outcomes.iter()
                    .map(|o| o.child)
                    .collect();
                ids.len() == outcome_count && ids.iter().filter(|id: &&NodeId| id.node_type() == NodeType::Terminal).count() <= 1
            },
            "AfterState outcomes must not contain duplicate node IDs and at most one terminal"
        );
    }
}
