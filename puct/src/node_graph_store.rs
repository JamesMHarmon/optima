use super::{AfterState, NodeArena, NodeGraph, NodeId, PUCTEdge, RollupStats, StateNode, Terminal};
use dashmap::DashMap;

/// Owns the node arena together with its transposition index.
///
/// This is the natural home for `hash -> NodeId` indexing since it is graph-wide state that must
/// stay consistent with the arena across pruning/compaction.
pub struct NodeGraphStore<A, R>
where
    R: RollupStats,
{
    arena: NodeArena<StateNode<A, R>, AfterState, Terminal<R>>,
    transposition_table: DashMap<u64, NodeId>,
}

impl<A, R> NodeGraphStore<A, R>
where
    R: RollupStats,
{
    pub fn new() -> Self {
        Self {
            arena: NodeArena::new(),
            transposition_table: DashMap::new(),
        }
    }

    #[inline]
    pub fn arena(&self) -> &NodeArena<StateNode<A, R>, AfterState, Terminal<R>> {
        &self.arena
    }

    #[inline]
    pub fn arena_mut(&mut self) -> &mut NodeArena<StateNode<A, R>, AfterState, Terminal<R>> {
        &mut self.arena
    }

    pub fn take_arena(&mut self) -> NodeArena<StateNode<A, R>, AfterState, Terminal<R>> {
        std::mem::replace(&mut self.arena, NodeArena::new())
    }

    pub fn set_arena(&mut self, arena: NodeArena<StateNode<A, R>, AfterState, Terminal<R>>) {
        self.arena = arena;
    }

    #[inline]
    pub fn graph(&self) -> NodeGraph<'_, A, R> {
        NodeGraph::new(&self.arena)
    }

    pub fn insert_transposition(&self, transposition_hash: u64, node_id: NodeId) -> Option<NodeId> {
        self.transposition_table.insert(transposition_hash, node_id)
    }

    pub fn get_transposition(&self, transposition_hash: u64) -> Option<NodeId> {
        self.transposition_table
            .get(&transposition_hash)
            .map(|v| *v)
    }

    pub fn prune_to_root(&mut self, root: NodeId) {
        let old_arena = self.take_arena();
        let rebuilt = super::rebuild_from_root(old_arena, root);

        self.set_arena(rebuilt.arena);
        self.rebuild_transpositions(rebuilt.transpositions);
    }

    pub fn prune_to_transposition_hash(&mut self, transposition_hash: u64) {
        let Some(root) = self.get_transposition(transposition_hash) else {
            self.set_arena(NodeArena::new());
            self.rebuild_transpositions(Vec::new());
            return;
        };

        self.prune_to_root(root);
    }

    pub fn rebuild_transpositions(&self, transpositions: Vec<(u64, NodeId)>) {
        self.transposition_table.clear();
        for (hash, node_id) in transpositions {
            self.transposition_table.insert(hash, node_id);
        }
    }

    /// Get child from edge if cached, otherwise lookup in transposition table and link.
    /// Returns None if this is a new position that needs expansion.
    pub fn get_or_link_transposition(
        &self,
        edge: &PUCTEdge,
        transposition_hash: u64,
    ) -> Option<NodeId> {
        let graph = self.graph();

        if let Some(nested_child_id) = graph.get_edge_state_with_hash(edge, transposition_hash) {
            return Some(nested_child_id);
        }

        if let Some(existing_id) = self.transposition_table.get(&transposition_hash) {
            graph.add_child_to_edge(edge, *existing_id);
            Some(*existing_id)
        } else {
            None
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
