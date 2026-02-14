use std::sync::atomic::{AtomicU32, Ordering};

use half::f16;
use model::ActionWithPolicy;
use tinyvec::TinyVec;

use crate::{
    AfterState, AfterStateOutcome, NodeArena, NodeGraph, PUCTEdge, RollupStats, StateNode,
    Terminal, WeightedMerge,
};

type TestStateNode = StateNode<u32, DummyRollup>;
type TestArena = NodeArena<TestStateNode, AfterState, Terminal<DummyRollup>>;

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

fn make_state_node(hash: u64) -> TestStateNode {
    let priors = [(0u32, 1.0f32)]
        .into_iter()
        .map(|(a, p)| ActionWithPolicy::new(a, f16::from_f32(p)))
        .collect::<Vec<_>>()
        .into_boxed_slice();

    StateNode::new(hash, priors, DummyRollup::default())
}

#[test]
fn get_edge_state_with_hash_increments_matching_after_state_outcome_visits() {
    let arena = TestArena::new();
    let graph = NodeGraph::new(&arena);

    let s0 = arena.push_state(make_state_node(100));
    let s1 = arena.push_state(make_state_node(200));

    let mut outcomes: TinyVec<[AfterStateOutcome; 2]> = TinyVec::new();
    outcomes.push(AfterStateOutcome::new(3, s0));
    outcomes.push(AfterStateOutcome::new(7, s1));
    let after_state_id = arena.push_after_state(AfterState::new(outcomes));

    let edge = PUCTEdge::new();
    edge.set_child(after_state_id);

    let before = arena
        .get_after_state_node(after_state_id)
        .outcomes
        .iter()
        .map(|o| (o.child(), o.visits()))
        .collect::<Vec<_>>();

    let found = graph.get_edge_state_with_hash(&edge, 200);
    assert_eq!(found, Some(s1));

    let after = arena
        .get_after_state_node(after_state_id)
        .outcomes
        .iter()
        .map(|o| (o.child(), o.visits()))
        .collect::<Vec<_>>();

    assert_eq!(before[0], after[0]);
    assert_eq!(after[1].0, s1);
    assert_eq!(after[1].1, before[1].1 + 1);
}

#[test]
fn add_child_to_edge_sets_new_outcome_visits_to_one() {
    let arena = TestArena::new();
    let graph = NodeGraph::new(&arena);

    let existing_child = arena.push_state(make_state_node(111));

    let edge = PUCTEdge::new();
    edge.increment_visits();
    assert!(edge.try_set_child(existing_child));

    let new_child = arena.push_state(make_state_node(222));

    edge.increment_visits();
    graph.add_child_to_edge(&edge, new_child);

    let after_state_id = edge.child().expect("edge child must be set");
    assert!(after_state_id.is_after_state());

    let after_state = arena.get_after_state_node(after_state_id);

    let new_outcome = after_state
        .outcomes
        .iter()
        .find(|o| o.child() == new_child)
        .expect("new outcome must exist");

    assert_eq!(new_outcome.visits(), 1);
}
