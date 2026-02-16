use super::{
    AfterState, AfterStateOutcome, NodeArena, NodeId, NodeType, PUCTEdge, RollupStats, StateNode,
    Terminal,
};

type GraphArena<A, R> = NodeArena<StateNode<A, R>, AfterState, Terminal<R>>;

/// Graph operations wrapper around NodeArena for node traversal and mutation.
pub struct NodeGraph<'a, A, R: RollupStats> {
    arena: &'a GraphArena<A, R>,
}

#[cfg(test)]
mod tests;

impl<'a, A, R: RollupStats> NodeGraph<'a, A, R> {
    pub fn new(arena: &'a GraphArena<A, R>) -> Self {
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
    pub fn find_edge_terminal(&self, edge: &PUCTEdge) -> Option<NodeId> {
        let child_id = edge.child()?;
        match child_id.node_type() {
            NodeType::Terminal => Some(child_id),
            NodeType::State => None,
            NodeType::AfterState => self
                .arena
                .get_after_state_node(child_id)
                .terminal_outcome()
                .map(|outcome| outcome.child()),
        }
    }

    /// If `edge` currently points to an AfterState, and that AfterState has a *direct* state
    /// outcome matching `transposition_hash`, increment that outcome's visits.
    ///
    /// Returns true if an outcome was found and incremented.
    pub fn increment_afterstate_visits(&self, edge: &PUCTEdge, transposition_hash: u64) -> bool {
        let after_state = match self.edge_after_state(edge) {
            None => return false,
            Some(after_state) => after_state,
        };

        after_state
            .outcomes
            .iter()
            .map(|outcome| (outcome, outcome.child()))
            .filter(|(_, child_id)| child_id.node_type() == NodeType::State)
            .find(|(_, child_id)| {
                self.arena.get_state_node(*child_id).transposition_hash() == transposition_hash
            })
            .is_some_and(|(outcome, _)| {
                outcome.increment_visits();
                true
            })
    }

    /// If `edge` currently points to an AfterState, and that AfterState has a terminal outcome,
    /// increment that terminal outcome's visits.
    ///
    /// Returns true if a terminal outcome existed and was incremented.
    pub fn increment_afterstate_terminal_visits(&self, edge: &PUCTEdge) -> bool {
        let after_state = match self.edge_after_state(edge) {
            None => return false,
            Some(after_state) => after_state,
        };

        match after_state.terminal_outcome() {
            None => false,
            Some(outcome) => {
                outcome.increment_visits();
                true
            }
        }
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

        new_outcomes.push(AfterStateOutcome::new(1, child_id));

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

    fn edge_after_state(&self, edge: &PUCTEdge) -> Option<&'a AfterState> {
        let child_id = edge.child()?;

        match child_id.node_type() {
            NodeType::AfterState => Some(self.arena.get_after_state_node(child_id)),
            _ => None,
        }
    }
}
