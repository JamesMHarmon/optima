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
    /// Optional helper for rebuilding a transposition table.
    pub transpositions: Vec<(u64, NodeId)>,
}

/// Stop-the-world compaction: rebuild a new arena containing only nodes reachable from `root`.
///
/// This consumes the old arena and produces a compact one, remapping all NodeIds and updating
/// edge/outcome child links accordingly.
///
/// Assumptions:
/// - No other threads are reading/writing the arena while this runs.
/// - No NodeIds/references from the old arena are used after this returns.
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
    old_states: Vec<Option<StateNode<A, R>>>,
    old_after_states: Vec<Option<AfterState>>,
    old_terminals: Vec<Option<Terminal<R>>>,

    live_states: Vec<bool>,
    live_after_states: Vec<bool>,
    live_terminals: Vec<bool>,

    state_map: Vec<Option<NodeId>>,
    after_state_map: Vec<Option<NodeId>>,
    terminal_map: Vec<Option<NodeId>>,

    new_state_ids: Vec<NodeId>,
    new_after_state_ids: Vec<NodeId>,

    new_arena: NodeArena<StateNode<A, R>, AfterState, Terminal<R>>,
}

impl<A, R> RebuildCtx<A, R>
where
    R: RollupStats,
{
    fn new(arena: NodeArena<StateNode<A, R>, AfterState, Terminal<R>>) -> Self {
        let (state_nodes, after_state_nodes, terminal_nodes) = arena.into_vecs();

        let old_states: Vec<Option<StateNode<A, R>>> = state_nodes.into_iter().map(Some).collect();
        let old_after_states: Vec<Option<AfterState>> =
            after_state_nodes.into_iter().map(Some).collect();
        let old_terminals: Vec<Option<Terminal<R>>> =
            terminal_nodes.into_iter().map(Some).collect();

        Self {
            live_states: vec![false; old_states.len()],
            live_after_states: vec![false; old_after_states.len()],
            live_terminals: vec![false; old_terminals.len()],

            state_map: vec![None; old_states.len()],
            after_state_map: vec![None; old_after_states.len()],
            terminal_map: vec![None; old_terminals.len()],

            old_states,
            old_after_states,
            old_terminals,

            new_state_ids: Vec::new(),
            new_after_state_ids: Vec::new(),

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
                    if self.live_states.get(idx).copied().unwrap_or(false) {
                        continue;
                    }
                    self.live_states[idx] = true;

                    let node = self.old_states[idx]
                        .as_ref()
                        .expect("live state node must exist");
                    for edge in node.iter_edges() {
                        if let Some(child) = edge.child() {
                            queue.push_back(child);
                        }
                    }
                }
                NodeType::AfterState => {
                    let idx = usize::from(id);
                    if self.live_after_states.get(idx).copied().unwrap_or(false) {
                        continue;
                    }
                    self.live_after_states[idx] = true;

                    let node = self.old_after_states[idx]
                        .as_ref()
                        .expect("live after-state node must exist");
                    for outcome in node.outcomes.iter() {
                        queue.push_back(outcome.child());
                    }
                }
                NodeType::Terminal => {
                    let idx = usize::from(id);
                    if self.live_terminals.get(idx).copied().unwrap_or(false) {
                        continue;
                    }
                    self.live_terminals[idx] = true;
                }
            }
        }
    }

    fn move_live_nodes(&mut self) {
        for (old_idx, is_live) in self.live_states.iter().copied().enumerate() {
            if !is_live {
                continue;
            }

            let node = self.old_states[old_idx]
                .take()
                .expect("live state node must exist");
            let new_id = self.new_arena.push_state(node);
            self.state_map[old_idx] = Some(new_id);
            self.new_state_ids.push(new_id);
        }

        for (old_idx, is_live) in self.live_after_states.iter().copied().enumerate() {
            if !is_live {
                continue;
            }

            let node = self.old_after_states[old_idx]
                .take()
                .expect("live after-state node must exist");
            let new_id = self.new_arena.push_after_state(node);
            self.after_state_map[old_idx] = Some(new_id);
            self.new_after_state_ids.push(new_id);
        }

        for (old_idx, is_live) in self.live_terminals.iter().copied().enumerate() {
            if !is_live {
                continue;
            }

            let node = self.old_terminals[old_idx]
                .take()
                .expect("live terminal node must exist");
            let new_id = self.new_arena.push_terminal(node);
            self.terminal_map[old_idx] = Some(new_id);
        }
    }

    fn remap_id(
        old: NodeId,
        state_map: &[Option<NodeId>],
        after_state_map: &[Option<NodeId>],
        terminal_map: &[Option<NodeId>],
    ) -> NodeId {
        if old.is_unset() {
            return NodeId::unset();
        }

        let old_idx = usize::from(old);
        match old.node_type() {
            NodeType::State => state_map[old_idx].expect("missing state remap"),
            NodeType::AfterState => after_state_map[old_idx].expect("missing after-state remap"),
            NodeType::Terminal => terminal_map[old_idx].expect("missing terminal remap"),
        }
    }

    fn patch_child_links(&mut self) {
        let (state_map, after_state_map, terminal_map) =
            (&self.state_map, &self.after_state_map, &self.terminal_map);
        let remap = |old: NodeId| Self::remap_id(old, state_map, after_state_map, terminal_map);

        // State edges
        for &state_id in &self.new_state_ids {
            let node = self.new_arena.get_state_node(state_id);
            for edge in node.iter_edges() {
                if let Some(child) = edge.child() {
                    edge.set_child(remap(child));
                }
            }
        }

        // AfterState outcomes
        for &after_state_id in &self.new_after_state_ids {
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
        let mut transpositions = Vec::with_capacity(self.new_state_ids.len());
        for &state_id in &self.new_state_ids {
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

#[cfg(test)]
mod tests;
