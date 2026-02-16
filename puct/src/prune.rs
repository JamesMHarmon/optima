use std::collections::VecDeque;

use super::{
    AfterState, AfterStateOutcome, NodeArena, NodeId, NodeType, RollupStats, StateNode, Terminal,
};

pub struct RebuiltArena<A, R>
where
    R: RollupStats,
{
    pub arena: NodeArena<StateNode<A, R>, AfterState, Terminal<R>>,
    pub root: NodeId,
    pub transpositions: Vec<(u64, NodeId)>,
}

/// Rebuild a new arena containing only nodes reachable from `root`.
///
/// This consumes the old arena and produces a compact one, remapping all NodeIds and updating
/// edge/outcome child links accordingly.
pub fn rebuild_from_root<A, R>(
    arena: NodeArena<StateNode<A, R>, AfterState, Terminal<R>>,
    root: NodeId,
) -> RebuiltArena<A, R>
where
    R: RollupStats,
{
    let mut ctx = RebuildCtx::new(arena);
    ctx.mark_reachable(root);
    ctx.move_live_nodes();
    ctx.patch_child_links();

    let transpositions = ctx.build_transpositions();
    let new_root = ctx.remap_root(root);

    RebuiltArena {
        arena: ctx.new_arena,
        root: new_root,
        transpositions,
    }
}

struct RebuildCtx<A, R>
where
    R: RollupStats,
{
    old_states: Vec<StateNode<A, R>>,
    old_after_states: Vec<AfterState>,
    old_terminals: Vec<Terminal<R>>,

    live_states: BitSet,
    live_after_states: BitSet,
    live_terminals: BitSet,

    state_map: Vec<NodeId>,
    after_state_map: Vec<NodeId>,
    terminal_map: Vec<NodeId>,

    new_arena: NodeArena<StateNode<A, R>, AfterState, Terminal<R>>,
}

impl<A, R> RebuildCtx<A, R>
where
    R: RollupStats,
{
    fn new(arena: NodeArena<StateNode<A, R>, AfterState, Terminal<R>>) -> Self {
        let (state_nodes, after_state_nodes, terminal_nodes) = arena.into_vecs();

        let old_states = state_nodes;
        let old_after_states = after_state_nodes;
        let old_terminals = terminal_nodes;

        Self {
            live_states: BitSet::new(old_states.len()),
            live_after_states: BitSet::new(old_after_states.len()),
            live_terminals: BitSet::new(old_terminals.len()),

            state_map: vec![NodeId::unset(); old_states.len()],
            after_state_map: vec![NodeId::unset(); old_after_states.len()],
            terminal_map: vec![NodeId::unset(); old_terminals.len()],

            old_states,
            old_after_states,
            old_terminals,

            new_arena: NodeArena::new(),
        }
    }

    fn mark_reachable(&mut self, root: NodeId) {
        let mut queue = VecDeque::new();
        queue.push_back(root);

        while let Some(id) = queue.pop_front() {
            if id.is_unset() {
                continue;
            }

            match id.node_type() {
                NodeType::State => {
                    let idx = usize::from(id);
                    if self.live_states.test_and_set(idx) {
                        continue;
                    }

                    let node = &self.old_states[idx];
                    for edge in node.iter_edges() {
                        if let Some(child) = edge.child() {
                            queue.push_back(child);
                        }
                    }
                }
                NodeType::AfterState => {
                    let idx = usize::from(id);
                    if self.live_after_states.test_and_set(idx) {
                        continue;
                    }

                    let node = &self.old_after_states[idx];
                    for outcome in node.outcomes.iter() {
                        queue.push_back(outcome.child());
                    }
                }
                NodeType::Terminal => {
                    let idx = usize::from(id);
                    if self.live_terminals.test_and_set(idx) {
                        continue;
                    }
                }
            }
        }
    }

    fn move_live_nodes(&mut self) {
        let old_states = std::mem::take(&mut self.old_states);
        for (old_idx, node) in old_states.into_iter().enumerate() {
            if !self.live_states.test(old_idx) {
                continue;
            }

            let new_id = self.new_arena.push_state(node);
            self.state_map[old_idx] = new_id;
        }

        let old_after_states = std::mem::take(&mut self.old_after_states);
        for (old_idx, node) in old_after_states.into_iter().enumerate() {
            if !self.live_after_states.test(old_idx) {
                continue;
            }

            let new_id = self.new_arena.push_after_state(node);
            self.after_state_map[old_idx] = new_id;
        }

        let old_terminals = std::mem::take(&mut self.old_terminals);
        for (old_idx, node) in old_terminals.into_iter().enumerate() {
            if !self.live_terminals.test(old_idx) {
                continue;
            }

            let new_id = self.new_arena.push_terminal(node);
            self.terminal_map[old_idx] = new_id;
        }
    }

    fn remap_id(
        old: NodeId,
        state_map: &[NodeId],
        after_state_map: &[NodeId],
        terminal_map: &[NodeId],
    ) -> NodeId {
        if old.is_unset() {
            return NodeId::unset();
        }

        let old_idx = usize::from(old);
        let new_id = match old.node_type() {
            NodeType::State => state_map[old_idx],
            NodeType::AfterState => after_state_map[old_idx],
            NodeType::Terminal => terminal_map[old_idx],
        };

        debug_assert!(!new_id.is_unset(), "missing remap for live node");
        new_id
    }

    fn patch_child_links(&mut self) {
        let (state_map, after_state_map, terminal_map) =
            (&self.state_map, &self.after_state_map, &self.terminal_map);
        let remap = |old: NodeId| Self::remap_id(old, state_map, after_state_map, terminal_map);

        // State edges
        for &state_id in &self.state_map {
            if state_id.is_unset() {
                continue;
            }
            let node = self.new_arena.get_state_node(state_id);
            for edge in node.iter_edges() {
                if let Some(child) = edge.child() {
                    edge.set_child(remap(child));
                }
            }
        }

        // AfterState outcomes
        for &after_state_id in &self.after_state_map {
            if after_state_id.is_unset() {
                continue;
            }
            let after_state = self.new_arena.get_after_state_node_mut(after_state_id);
            for outcome in after_state.outcomes.iter_mut() {
                let child = outcome.child();
                let visits = outcome.visits();
                *outcome = AfterStateOutcome::new(visits, remap(child));
            }

            debug_assert!(
                after_state.is_valid(),
                "AfterState outcomes must remain valid after remap"
            );
        }
    }

    fn build_transpositions(&self) -> Vec<(u64, NodeId)> {
        let mut transpositions = Vec::new();
        for &state_id in &self.state_map {
            if state_id.is_unset() {
                continue;
            }
            let node = self.new_arena.get_state_node(state_id);
            transpositions.push((node.transposition_hash(), state_id));
        }
        transpositions
    }

    fn remap_root(&self, root: NodeId) -> NodeId {
        if root.is_unset() {
            root
        } else {
            Self::remap_id(
                root,
                &self.state_map,
                &self.after_state_map,
                &self.terminal_map,
            )
        }
    }
}

struct BitSet {
    bits: Vec<u64>,
}

impl BitSet {
    fn new(len: usize) -> Self {
        let words = len.div_ceil(64);
        Self {
            bits: vec![0; words],
        }
    }

    #[inline]
    fn test(&self, index: usize) -> bool {
        let word = index / 64;
        let bit = index % 64;
        (self.bits[word] >> bit) & 1 == 1
    }

    /// Returns previous value.
    #[inline]
    fn test_and_set(&mut self, index: usize) -> bool {
        let word = index / 64;
        let bit = index % 64;
        let mask = 1u64 << bit;
        let was_set = (self.bits[word] & mask) != 0;
        self.bits[word] |= mask;
        was_set
    }
}

#[cfg(test)]
mod tests;
