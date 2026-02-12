use std::collections::VecDeque;

use super::{
    AfterState, AfterStateOutcome, NodeArena, NodeId, NodeType, RollupStats, StateNode, Terminal,
};

pub struct RebuiltArena<A, R, SI>
where
    R: RollupStats,
{
    pub arena: NodeArena<StateNode<A, R, SI>, AfterState, Terminal<R>>,
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
pub fn rebuild_from_root<A, R, SI>(
    arena: NodeArena<StateNode<A, R, SI>, AfterState, Terminal<R>>,
    root: NodeId,
) -> RebuiltArena<A, R, SI>
where
    R: RollupStats,
{
    let (state_nodes, after_state_nodes, terminal_nodes) = arena.into_vecs();

    // 1) Mark reachable nodes.
    let mut live_states = vec![false; state_nodes.len()];
    let mut live_after_states = vec![false; after_state_nodes.len()];
    let mut live_terminals = vec![false; terminal_nodes.len()];

    let mut queue = VecDeque::new();
    queue.push_back(root);

    while let Some(id) = queue.pop_front() {
        if id.is_unset() {
            continue;
        }

        match id.node_type() {
            NodeType::State => {
                let idx = usize::from(id);
                if live_states.get(idx).copied().unwrap_or(false) {
                    continue;
                }
                live_states[idx] = true;

                let node = &state_nodes[idx];
                for edge in node.iter_edges() {
                    if let Some(child) = edge.child() {
                        queue.push_back(child);
                    }
                }
            }
            NodeType::AfterState => {
                let idx = usize::from(id);
                if live_after_states.get(idx).copied().unwrap_or(false) {
                    continue;
                }
                live_after_states[idx] = true;

                let node = &after_state_nodes[idx];
                for outcome in node.outcomes.iter() {
                    queue.push_back(outcome.child());
                }
            }
            NodeType::Terminal => {
                let idx = usize::from(id);
                if live_terminals.get(idx).copied().unwrap_or(false) {
                    continue;
                }
                live_terminals[idx] = true;
            }
        }
    }

    // 2) Move live nodes into a new arena, recording old->new ID mappings.
    let mut old_states: Vec<Option<StateNode<A, R, SI>>> =
        state_nodes.into_iter().map(Some).collect();
    let mut old_after_states: Vec<Option<AfterState>> =
        after_state_nodes.into_iter().map(Some).collect();
    let mut old_terminals: Vec<Option<Terminal<R>>> =
        terminal_nodes.into_iter().map(Some).collect();

    let mut new_arena: NodeArena<StateNode<A, R, SI>, AfterState, Terminal<R>> = NodeArena::new();

    let mut state_map: Vec<Option<NodeId>> = vec![None; old_states.len()];
    let mut after_state_map: Vec<Option<NodeId>> = vec![None; old_after_states.len()];
    let mut terminal_map: Vec<Option<NodeId>> = vec![None; old_terminals.len()];

    let mut new_state_ids = Vec::new();
    let mut new_after_state_ids = Vec::new();
    let mut new_terminal_ids = Vec::new();

    // Ensure the root (state node) is first so callers can keep using "root == 0" conventions.
    if !root.is_unset() {
        debug_assert!(
            root.is_state(),
            "rebuild_from_root currently expects a State root"
        );
        let root_idx = usize::from(root);
        debug_assert!(live_states.get(root_idx).copied().unwrap_or(false));

        if let Some(node) = old_states[root_idx].take() {
            let new_id = new_arena.push_state(node);
            state_map[root_idx] = Some(new_id);
            new_state_ids.push(new_id);
        }
    }

    for (old_idx, is_live) in live_states.iter().copied().enumerate() {
        if !is_live {
            continue;
        }
        if !root.is_unset() && old_idx == usize::from(root) {
            continue;
        }

        let node = old_states[old_idx]
            .take()
            .expect("live state node must exist");
        let new_id = new_arena.push_state(node);
        state_map[old_idx] = Some(new_id);
        new_state_ids.push(new_id);
    }

    for (old_idx, is_live) in live_after_states.iter().copied().enumerate() {
        if !is_live {
            continue;
        }

        let node = old_after_states[old_idx]
            .take()
            .expect("live after-state node must exist");
        let new_id = new_arena.push_after_state(node);
        after_state_map[old_idx] = Some(new_id);
        new_after_state_ids.push(new_id);
    }

    for (old_idx, is_live) in live_terminals.iter().copied().enumerate() {
        if !is_live {
            continue;
        }

        let node = old_terminals[old_idx]
            .take()
            .expect("live terminal node must exist");
        let new_id = new_arena.push_terminal(node);
        terminal_map[old_idx] = Some(new_id);
        new_terminal_ids.push(new_id);
    }

    let remap = |old: NodeId,
                 state_map: &Vec<Option<NodeId>>,
                 after_state_map: &Vec<Option<NodeId>>,
                 terminal_map: &Vec<Option<NodeId>>| {
        if old.is_unset() {
            return NodeId::unset();
        }
        let old_idx = usize::from(old);
        match old.node_type() {
            NodeType::State => state_map[old_idx].expect("missing state remap"),
            NodeType::AfterState => after_state_map[old_idx].expect("missing after-state remap"),
            NodeType::Terminal => terminal_map[old_idx].expect("missing terminal remap"),
        }
    };

    // 3) Patch child links to use new NodeIds.
    // State edges
    for &state_id in &new_state_ids {
        let node = new_arena.get_state_node(state_id);
        for edge in node.iter_edges() {
            if let Some(child) = edge.child() {
                let new_child = remap(child, &state_map, &after_state_map, &terminal_map);
                edge.set_child(new_child);
            }
        }
    }

    // AfterState outcomes
    for &after_state_id in &new_after_state_ids {
        let after_state = new_arena.get_after_state_node_mut(after_state_id);
        for outcome in after_state.outcomes.iter_mut() {
            let child = outcome.child();
            let visits = outcome.visits();
            let new_child = remap(child, &state_map, &after_state_map, &terminal_map);
            *outcome = AfterStateOutcome::new(visits, new_child);
        }

        debug_assert!(
            after_state.is_valid(),
            "AfterState outcomes must remain valid after remap"
        );
    }

    // 4) Provide transposition entries for the new arena.
    let mut transpositions = Vec::with_capacity(new_state_ids.len());
    for &state_id in &new_state_ids {
        let node = new_arena.get_state_node(state_id);
        transpositions.push((node.transposition_hash(), state_id));
    }

    let new_root = if root.is_unset() {
        root
    } else {
        state_map[usize::from(root)].expect("root must be remapped")
    };

    // Sanity: if caller uses the "root == 0" convention, we preserved it by inserting root first.
    if !new_root.is_unset() {
        debug_assert_eq!(usize::from(new_root), 0);
    }

    RebuiltArena {
        arena: new_arena,
        root: new_root,
        transpositions,
    }
}
