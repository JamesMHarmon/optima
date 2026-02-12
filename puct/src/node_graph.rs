use super::{
    AfterState, AfterStateOutcome, NodeArena, NodeId, NodeType, PUCTEdge, RollupStats, StateNode,
    Terminal,
};

/// Graph operations wrapper around NodeArena for node traversal and mutation.
pub struct NodeGraph<'a, A, R: RollupStats, SI> {
    arena: &'a NodeArena<StateNode<A, R, SI>, AfterState, Terminal<R>>,
}

impl<'a, A, R: RollupStats, SI> NodeGraph<'a, A, R, SI> {
    pub fn new(arena: &'a NodeArena<StateNode<A, R, SI>, AfterState, Terminal<R>>) -> Self {
        Self { arena }
    }

    /// Find a State node with matching transposition hash, traversing through AfterState nodes if needed.
    pub fn find_state_with_hash(&self, node_id: NodeId, transposition_hash: u64) -> Option<NodeId> {
        match node_id.node_type() {
            NodeType::State => {
                let state = self.arena.get_state_node(node_id);
                (state.transposition_hash() == transposition_hash).then_some(node_id)
            }
            NodeType::AfterState => {
                let after_state = self.arena.get_after_state_node(node_id);
                after_state.outcomes.iter().find_map(|outcome| {
                    self.find_state_with_hash(outcome.child(), transposition_hash)
                })
            }
            NodeType::Terminal => None,
        }
    }

    /// Check if an edge already points to a state with the given transposition hash.
    pub fn get_edge_state_with_hash(
        &self,
        edge: &PUCTEdge,
        transposition_hash: u64,
    ) -> Option<NodeId> {
        edge.child()
            .and_then(|child_id| self.find_state_with_hash(child_id, transposition_hash))
    }

    /// Find terminal node reachable through edge and return its ID with visit count.
    /// Returns edge visits if pointing directly to terminal, or outcome visits if through AfterState.
    pub fn find_edge_terminal(&self, edge: &PUCTEdge) -> Option<(NodeId, u32)> {
        edge.child()
            .and_then(|child_id| match child_id.node_type() {
                NodeType::Terminal => Some((child_id, edge.visits())),

                NodeType::AfterState => self
                    .arena
                    .get_after_state_node(child_id)
                    .terminal_outcome()
                    .map(|outcome| outcome.as_tuple()),
                NodeType::State => None,
            })
    }

    /// Add a child to an edge, converting to AfterState if multiple outcomes exist.
    pub fn add_child_to_edge(&self, edge: &PUCTEdge, child_id: NodeId) {
        if edge.try_set_child(child_id) {
            return;
        }

        let existing_child_id = edge
            .child()
            .expect("Child must be set if try_set_child failed");

        let mut new_outcomes = tinyvec::TinyVec::new();

        match existing_child_id.node_type() {
            NodeType::AfterState => {
                let after_state = self.arena.get_after_state_node(existing_child_id);
                for outcome in &after_state.outcomes {
                    new_outcomes.push(outcome.clone());
                }
            }
            NodeType::State | NodeType::Terminal => {
                new_outcomes.push(AfterStateOutcome::new(edge.visits(), existing_child_id));
            }
        }

        // @TODO: Should this be visit of 1?
        new_outcomes.push(AfterStateOutcome::new(0, child_id));

        // Create new AfterState and atomically update edge
        let new_after_state_id = self.arena.push_after_state(AfterState::new(new_outcomes));
        edge.set_child(new_after_state_id);

        debug_assert!(
            self.arena
                .get_after_state_node(new_after_state_id)
                .is_valid(),
            "AfterState outcomes must not contain duplicate node IDs and at most one terminal"
        );
    }
}
