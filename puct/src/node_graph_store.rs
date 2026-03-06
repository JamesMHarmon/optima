use crate::after_state::AfterState;
use crate::edge::PUCTEdge;
use crate::node::StateNode;
use crate::node_arena::{NodeArena, NodeId};
use crate::node_graph::NodeGraph;
use crate::prune::rebuild_from_root;
use crate::rollup::RollupStats;
use crate::selection_policy::EdgeInfo;
use crate::terminal_node::Terminal;
use dashmap::{DashMap, DashSet};
use model::ActionWithPolicy;

/// Owns the node arena together with its transposition index.
pub(super) struct NodeGraphStore<A, R>
where
    R: RollupStats,
{
    arena: NodeArena<StateNode<A, R>, AfterState, Terminal<R>>,
    transposition_table: DashMap<u64, NodeId>,
    /// Hashes claimed by a sim that is currently awaiting neural-net expansion.
    pending: DashSet<u64>,
}

impl<A, R> NodeGraphStore<A, R>
where
    R: RollupStats,
{
    pub(super) fn new() -> Self {
        Self {
            arena: NodeArena::new(),
            transposition_table: DashMap::new(),
            pending: DashSet::new(),
        }
    }

    #[inline]
    pub(super) fn graph(&self) -> NodeGraph<'_, A, R> {
        NodeGraph::new(&self.arena)
    }

    #[inline]
    pub(super) fn state_node(&self, node_id: NodeId) -> &StateNode<A, R> {
        self.arena.state_node(node_id)
    }

    #[inline]
    pub(super) fn state_node_mut(&mut self, node_id: NodeId) -> &mut StateNode<A, R> {
        self.arena.state_node_mut(node_id)
    }

    #[inline]
    pub(super) fn terminal_node(&self, node_id: NodeId) -> &Terminal<R> {
        self.arena.terminal_node(node_id)
    }

    #[inline]
    pub(super) fn create_and_insert_state_node(
        &self,
        transposition_hash: u64,
        policy_priors: Vec<ActionWithPolicy<A>>,
        rollup_stats: R,
    ) -> NodeId {
        debug_assert!(
            !policy_priors.is_empty(),
            "Cannot create state node without actions - should be terminal"
        );

        let node = StateNode::new(transposition_hash, policy_priors, rollup_stats);
        let new_node_id = self.arena.push_state(node);

        let previous_entry = self
            .transposition_table
            .insert(transposition_hash, new_node_id);

        debug_assert!(
            previous_entry.is_none(),
            "Transposition table entry for hash already exists"
        );

        // Release the pending claim now that the node is in the table.
        self.pending.remove(&transposition_hash);

        new_node_id
    }

    #[inline]
    pub(super) fn create_and_insert_terminal_node(&self, rollup_stats: R) -> NodeId {
        self.arena.push_terminal(Terminal::new(rollup_stats))
    }

    #[inline]
    pub(super) fn recompute_rollup(&self, node_id: NodeId) {
        let node = self.arena.state_node(node_id);
        node.recompute_rollup(&self.arena);
    }

    #[inline]
    pub(super) fn iter_edge_info<'a>(
        &'a self,
        node: &'a StateNode<A, R>,
    ) -> impl Iterator<Item = EdgeInfo<'a, A, R::Snapshot>> + 'a {
        node.iter_edge_info(&self.arena)
    }

    pub(super) fn prune_to_transposition_hash(&mut self, transposition_hash: u64) {
        let Some(root) = self
            .transposition_table
            .get(&transposition_hash)
            .map(|v| *v)
        else {
            self.arena = NodeArena::new();
            self.rebuild_transpositions(Vec::new());
            return;
        };

        self.prune_to_root(root);
    }

    #[inline]
    pub(super) fn get_node_id(&self, transposition_hash: u64) -> Option<NodeId> {
        self.transposition_table
            .get(&transposition_hash)
            .map(|v| *v)
    }

    /// Determine what a sim should do upon reaching a position with `transposition_hash`.
    ///
    /// - [`LeafResult::Known`]: the node is already in the tree; keep traversing.
    /// - [`LeafResult::Claimed`]: this sim won the atomic claim; it must send
    ///   `SimMsg::State` and call [`create_and_insert_state_node`] once analysis
    ///   completes (which releases the claim).
    /// - [`LeafResult::Preempted`]: another sim already claimed this hash;
    ///   this sim should send `SimMsg::Preempted` so backprop removes its virtual loss
    ///   in order.
    pub(super) fn link_or_try_claim(&self, edge: &PUCTEdge, transposition_hash: u64) -> LeafResult {
        let graph = self.graph();

        if let Some(nested_child_id) = graph.get_edge_state_with_hash(edge, transposition_hash) {
            return LeafResult::Known(nested_child_id);
        }

        if let Some(existing_id) = self.transposition_table.get(&transposition_hash) {
            // Node is in the tree; link edge and keep traversing.
            let _ = edge.try_set_child(*existing_id);
            return LeafResult::Known(*existing_id);
        }

        // Position is genuinely new. Race to claim it.
        if self.pending.insert(transposition_hash) {
            LeafResult::Claimed
        } else {
            LeafResult::Preempted
        }
    }

    fn prune_to_root(&mut self, root: NodeId) {
        let old_arena = std::mem::replace(&mut self.arena, NodeArena::new());
        let rebuilt = rebuild_from_root(old_arena, root);

        self.arena = rebuilt.arena;
        self.rebuild_transpositions(rebuilt.transpositions);
    }

    fn rebuild_transpositions(&self, transpositions: Vec<(u64, NodeId)>) {
        self.transposition_table.clear();
        self.pending.clear();
        for (hash, node_id) in transpositions {
            self.transposition_table.insert(hash, node_id);
        }
    }
}

impl<A, R> Default for NodeGraphStore<A, R>
where
    R: RollupStats,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Outcome of [`NodeGraphStore::link_or_try_claim`].
pub(super) enum LeafResult {
    /// The node is already in the tree. The edge has been linked.
    Known(NodeId),
    /// This sim is the first to reach this hash — it should send `SimMsg::State`.
    Claimed,
    /// Another sim already claimed this hash and will expand it; this sim is preempted.
    Preempted,
}

#[cfg(test)]
mod tests;
