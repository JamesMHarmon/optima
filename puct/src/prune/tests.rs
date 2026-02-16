use std::sync::atomic::{AtomicU32, Ordering};

use half::f16;
use model::ActionWithPolicy;

use crate::{AfterState, NodeArena, RollupStats, StateNode, Terminal, WeightedMerge};

use super::rebuild_from_root;

#[derive(Default)]
struct DummyRollup {
    v: AtomicU32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct DummySnapshot(u32);

impl WeightedMerge for DummySnapshot {
    fn zero() -> Self {
        Self(0)
    }

    fn merge_weighted(&mut self, other: &Self, weight: u32) {
        self.0 = self.0.wrapping_add(other.0.wrapping_mul(weight));
    }
}

impl RollupStats for DummyRollup {
    type Snapshot = DummySnapshot;

    fn snapshot(&self) -> Self::Snapshot {
        DummySnapshot(self.v.load(Ordering::Relaxed))
    }

    fn set(&self, value: Self::Snapshot) {
        self.v.store(value.0, Ordering::Relaxed);
    }
}

fn make_node(hash: u64) -> StateNode<u32, DummyRollup> {
    let priors: Box<[ActionWithPolicy<u32>]> =
        vec![ActionWithPolicy::new(0, f16::from_f32(1.0))].into_boxed_slice();
    StateNode::new(hash, priors, DummyRollup::default())
}

#[test]
fn rebuild_from_root_does_not_force_root_to_zero() {
    type Arena = NodeArena<StateNode<u32, DummyRollup>, AfterState, Terminal<DummyRollup>>;

    let arena: Arena = NodeArena::new();
    let state0 = arena.push_state(make_node(0));
    let state1 = arena.push_state(make_node(1));

    // Make both nodes reachable from root=state1 by linking state1 -> state0.
    let root_node = arena.get_state_node(state1);
    root_node.ensure_frontier_edge();
    let (edge, _) = root_node.edge_and_action(0);
    edge.set_child(state0);

    let rebuilt = rebuild_from_root(arena, state1);

    // With the natural old-index insertion order, state0 becomes new idx 0 and root (old idx 1)
    // becomes new idx 1.
    assert_eq!(usize::from(rebuilt.root), 1);

    let (states, _after_states, _terminals) = rebuilt.arena.into_vecs();
    assert_eq!(states.len(), 2);

    let rebuilt_root = &states[1];
    let child = rebuilt_root.iter_edges().next().unwrap().child().unwrap();
    assert_eq!(usize::from(child), 0);
}
