use std::sync::atomic::{AtomicU32, Ordering};

use half::f16;
use model::ActionWithPolicy;
use tinyvec::TinyVec;

use crate::after_state::{AfterState, AfterStateOutcome};
use crate::edge::PUCTEdge;
use crate::node::StateNode;
use crate::node_arena::{NodeArena, NodeId};
use crate::node_graph::NodeGraph;
use crate::rollup::{RollupStats, WeightedMerge};
use crate::terminal_node::Terminal;

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

fn make_terminal_node(arena: &TestArena, value: u32) -> NodeId {
    let r = DummyRollup::default();
    r.set(DummySnapshot(value));
    arena.push_terminal(Terminal::new(r))
}

fn after_state_outcome_visits(arena: &TestArena, after_state_id: NodeId) -> Vec<(NodeId, u32)> {
    arena
        .get_after_state_node(after_state_id)
        .outcomes
        .iter()
        .map(|o| (o.child(), o.visits()))
        .collect::<Vec<_>>()
}

#[test]
fn terminal_merge_accumulates_when_terminal_already_reachable_via_edge() {
    let arena = TestArena::new();
    let graph = NodeGraph::new(&arena);

    let terminal_id = make_terminal_node(&arena, 10);

    let edge = PUCTEdge::new();
    edge.set_child(terminal_id);

    let found_terminal_id = graph
        .find_edge_terminal(&edge)
        .expect("terminal must be reachable");
    assert_eq!(found_terminal_id, terminal_id);

    let terminal_node = arena.get_terminal_node(found_terminal_id);
    assert_eq!(terminal_node.rollup_stats().snapshot(), DummySnapshot(10));

    terminal_node.rollup_stats().accumulate(&DummySnapshot(7));
    assert_eq!(terminal_node.rollup_stats().snapshot(), DummySnapshot(17));
}

#[test]
fn get_edge_state_with_hash_returns_none_when_edge_has_no_child() {
    let arena = TestArena::new();
    let graph = NodeGraph::new(&arena);
    let edge = PUCTEdge::new();

    assert_eq!(edge.child(), None);
    assert_eq!(graph.get_edge_state_with_hash(&edge, 123), None);
}

#[test]
fn get_edge_state_with_hash_matches_state_child_without_touching_edge_visits() {
    let arena = TestArena::new();
    let graph = NodeGraph::new(&arena);
    let s0 = arena.push_state(make_state_node(42));

    let edge = PUCTEdge::new();
    edge.increment_visits();
    edge.increment_visits();
    let edge_visits_before = edge.visits();
    assert!(edge.try_set_child(s0));

    assert_eq!(graph.get_edge_state_with_hash(&edge, 42), Some(s0));
    assert_eq!(edge.visits(), edge_visits_before);
}

#[test]
fn get_edge_state_with_hash_state_child_mismatch_returns_none() {
    let arena = TestArena::new();
    let graph = NodeGraph::new(&arena);
    let s0 = arena.push_state(make_state_node(1));
    let edge = PUCTEdge::new();
    assert!(edge.try_set_child(s0));

    assert_eq!(graph.get_edge_state_with_hash(&edge, 2), None);
}

#[test]
fn get_edge_state_with_hash_terminal_child_returns_none() {
    let arena = TestArena::new();
    let graph = NodeGraph::new(&arena);
    let t0 = make_terminal_node(&arena, 9);
    let edge = PUCTEdge::new();
    assert!(edge.try_set_child(t0));

    assert_eq!(graph.get_edge_state_with_hash(&edge, 123), None);
}

#[test]
fn get_edge_state_with_hash_after_state_no_match_does_not_increment_any_outcome() {
    let arena = TestArena::new();
    let graph = NodeGraph::new(&arena);

    let s0 = arena.push_state(make_state_node(10));
    let s1 = arena.push_state(make_state_node(20));

    let mut outcomes: TinyVec<[AfterStateOutcome; 2]> = TinyVec::new();
    outcomes.push(AfterStateOutcome::new(2, s0));
    outcomes.push(AfterStateOutcome::new(3, s1));
    let after_state_id = arena.push_after_state(AfterState::new(outcomes));

    let edge = PUCTEdge::new();
    edge.set_child(after_state_id);

    let before = after_state_outcome_visits(&arena, after_state_id);
    assert_eq!(graph.get_edge_state_with_hash(&edge, 999), None);
    let after = after_state_outcome_visits(&arena, after_state_id);

    assert_eq!(before, after);
}

#[test]
fn get_edge_state_with_hash_after_state_ignores_terminal_outcome() {
    let arena = TestArena::new();
    let graph = NodeGraph::new(&arena);

    let s0 = arena.push_state(make_state_node(111));
    let t0 = make_terminal_node(&arena, 5);

    let mut outcomes: TinyVec<[AfterStateOutcome; 2]> = TinyVec::new();
    outcomes.push(AfterStateOutcome::new(4, t0));
    outcomes.push(AfterStateOutcome::new(6, s0));
    let after_state_id = arena.push_after_state(AfterState::new(outcomes));

    let edge = PUCTEdge::new();
    edge.set_child(after_state_id);

    let before = after_state_outcome_visits(&arena, after_state_id);
    assert!(graph.increment_afterstate_visits(&edge, 111));
    assert_eq!(graph.get_edge_state_with_hash(&edge, 111), Some(s0));
    let after = after_state_outcome_visits(&arena, after_state_id);

    // Only the state outcome increments.
    assert_eq!(after[0], before[0]);
    assert_eq!(after[1].0, s0);
    assert_eq!(after[1].1, before[1].1 + 1);
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

    assert!(graph.increment_afterstate_visits(&edge, 200));
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

#[test]
fn find_edge_terminal_direct_terminal_does_not_mutate_edge_visits() {
    let arena = TestArena::new();
    let graph = NodeGraph::new(&arena);

    let t0 = make_terminal_node(&arena, 9);
    let edge = PUCTEdge::new();
    edge.increment_visits();
    edge.increment_visits();
    let edge_visits_before = edge.visits();
    assert!(edge.try_set_child(t0));

    assert_eq!(graph.find_edge_terminal(&edge), Some(t0));
    assert_eq!(edge.visits(), edge_visits_before);
}

#[test]
fn find_edge_terminal_after_state_increments_terminal_outcome_only() {
    let arena = TestArena::new();
    let graph = NodeGraph::new(&arena);

    let s0 = arena.push_state(make_state_node(1));
    let t0 = make_terminal_node(&arena, 99);

    let mut outcomes: TinyVec<[AfterStateOutcome; 2]> = TinyVec::new();
    outcomes.push(AfterStateOutcome::new(10, s0));
    outcomes.push(AfterStateOutcome::new(20, t0));
    let after_state_id = arena.push_after_state(AfterState::new(outcomes));

    let edge = PUCTEdge::new();
    edge.set_child(after_state_id);

    let before = after_state_outcome_visits(&arena, after_state_id);
    assert!(graph.increment_afterstate_terminal_visits(&edge));
    assert_eq!(graph.find_edge_terminal(&edge), Some(t0));
    let after = after_state_outcome_visits(&arena, after_state_id);

    // Only terminal outcome increments.
    assert_eq!(after[0], before[0]);
    assert_eq!(after[1].0, t0);
    assert_eq!(after[1].1, before[1].1 + 1);

    // Repeated calls continue to increment terminal outcome.
    assert!(graph.increment_afterstate_terminal_visits(&edge));
    assert_eq!(graph.find_edge_terminal(&edge), Some(t0));
    let after2 = after_state_outcome_visits(&arena, after_state_id);
    assert_eq!(after2[1].1, after[1].1 + 1);
}

#[test]
fn find_edge_terminal_after_state_without_terminal_returns_none_and_does_not_increment() {
    let arena = TestArena::new();
    let graph = NodeGraph::new(&arena);

    let s0 = arena.push_state(make_state_node(1));
    let s1 = arena.push_state(make_state_node(2));

    let mut outcomes: TinyVec<[AfterStateOutcome; 2]> = TinyVec::new();
    outcomes.push(AfterStateOutcome::new(2, s0));
    outcomes.push(AfterStateOutcome::new(3, s1));
    let after_state_id = arena.push_after_state(AfterState::new(outcomes));

    let edge = PUCTEdge::new();
    edge.set_child(after_state_id);

    let before = after_state_outcome_visits(&arena, after_state_id);
    assert_eq!(graph.find_edge_terminal(&edge), None);
    let after = after_state_outcome_visits(&arena, after_state_id);
    assert_eq!(before, after);
}

#[test]
fn add_child_to_edge_when_unset_sets_child_directly_without_after_state() {
    let arena = TestArena::new();
    let graph = NodeGraph::new(&arena);

    let child = arena.push_state(make_state_node(1));
    let edge = PUCTEdge::new();
    assert_eq!(edge.child(), None);

    graph.add_child_to_edge(&edge, child);

    let got = edge.child().expect("child must be set");
    assert_eq!(got, child);
    assert!(got.is_state());
}

#[test]
fn add_child_to_edge_converts_state_child_to_after_state_and_preserves_existing_visits() {
    let arena = TestArena::new();
    let graph = NodeGraph::new(&arena);

    let existing = arena.push_state(make_state_node(100));
    let new_child = arena.push_state(make_state_node(200));

    let edge = PUCTEdge::new();
    for _ in 0..5 {
        edge.increment_visits();
    }
    let edge_visits_before = edge.visits();
    assert!(edge.try_set_child(existing));

    graph.add_child_to_edge(&edge, new_child);

    assert_eq!(edge.visits(), edge_visits_before);
    let after_state_id = edge.child().expect("child must exist");
    assert!(after_state_id.is_after_state());

    let after = arena.get_after_state_node(after_state_id);
    assert_eq!(after.outcomes.len(), 2);

    let existing_outcome = after
        .outcomes
        .iter()
        .find(|o| o.child() == existing)
        .expect("existing outcome must exist");
    assert_eq!(existing_outcome.visits(), edge_visits_before);

    let new_outcome = after
        .outcomes
        .iter()
        .find(|o| o.child() == new_child)
        .expect("new outcome must exist");
    assert_eq!(new_outcome.visits(), 1);
}

#[test]
fn add_child_to_edge_converts_terminal_child_to_after_state_and_preserves_existing_visits() {
    let arena = TestArena::new();
    let graph = NodeGraph::new(&arena);

    let existing = make_terminal_node(&arena, 9);
    let new_child = arena.push_state(make_state_node(200));

    let edge = PUCTEdge::new();
    for _ in 0..3 {
        edge.increment_visits();
    }
    let edge_visits_before = edge.visits();
    assert!(edge.try_set_child(existing));

    graph.add_child_to_edge(&edge, new_child);

    assert_eq!(edge.visits(), edge_visits_before);
    let after_state_id = edge.child().expect("child must exist");
    assert!(after_state_id.is_after_state());

    let after = arena.get_after_state_node(after_state_id);
    assert_eq!(after.outcomes.len(), 2);

    let existing_outcome = after
        .outcomes
        .iter()
        .find(|o| o.child() == existing)
        .expect("existing outcome must exist");
    assert_eq!(existing_outcome.visits(), edge_visits_before);

    let new_outcome = after
        .outcomes
        .iter()
        .find(|o| o.child() == new_child)
        .expect("new outcome must exist");
    assert_eq!(new_outcome.visits(), 1);
}

#[test]
fn add_child_to_edge_when_existing_child_is_after_state_clones_outcomes_without_reweighting() {
    let arena = TestArena::new();
    let graph = NodeGraph::new(&arena);

    let s0 = arena.push_state(make_state_node(1));
    let s1 = arena.push_state(make_state_node(2));
    let new_child = arena.push_state(make_state_node(3));

    let mut outcomes: TinyVec<[AfterStateOutcome; 2]> = TinyVec::new();
    outcomes.push(AfterStateOutcome::new(2, s0));
    outcomes.push(AfterStateOutcome::new(9, s1));
    let after_state_id = arena.push_after_state(AfterState::new(outcomes));

    let edge = PUCTEdge::new();
    for _ in 0..100 {
        edge.increment_visits();
    }
    edge.set_child(after_state_id);

    let before = after_state_outcome_visits(&arena, after_state_id);
    graph.add_child_to_edge(&edge, new_child);

    let new_after_state_id = edge.child().expect("child must exist");
    assert!(new_after_state_id.is_after_state());

    // Original afterstate stays unchanged.
    let after_original = after_state_outcome_visits(&arena, after_state_id);
    assert_eq!(before, after_original);

    // New afterstate contains old outcomes unchanged + new outcome at 1.
    let after_new = after_state_outcome_visits(&arena, new_after_state_id);
    assert!(after_new.iter().any(|(id, v)| *id == s0 && *v == 2));
    assert!(after_new.iter().any(|(id, v)| *id == s1 && *v == 9));
    assert!(after_new.iter().any(|(id, v)| *id == new_child && *v == 1));
}

#[test]
fn find_state_with_hash_terminal_root_returns_none() {
    let arena = TestArena::new();
    let graph = NodeGraph::new(&arena);

    let t0 = make_terminal_node(&arena, 1);
    assert_eq!(graph.find_state_with_hash(t0, 123), None);
}

#[test]
fn find_state_with_hash_state_root_matches_and_mismatches() {
    let arena = TestArena::new();
    let graph = NodeGraph::new(&arena);

    let s0 = arena.push_state(make_state_node(999));
    assert_eq!(graph.find_state_with_hash(s0, 999), Some(s0));
    assert_eq!(graph.find_state_with_hash(s0, 1000), None);
}

#[test]
fn find_state_with_hash_after_state_root_finds_state_depth_first() {
    let arena = TestArena::new();
    let graph = NodeGraph::new(&arena);

    let s0 = arena.push_state(make_state_node(10));
    let s1 = arena.push_state(make_state_node(20));

    let mut outcomes: TinyVec<[AfterStateOutcome; 2]> = TinyVec::new();
    outcomes.push(AfterStateOutcome::new(1, s0));
    outcomes.push(AfterStateOutcome::new(1, s1));
    let after_state_id = arena.push_after_state(AfterState::new(outcomes));

    assert_eq!(graph.find_state_with_hash(after_state_id, 20), Some(s1));
}

#[test]
fn find_state_with_hash_does_not_mutate_after_state_outcome_visits() {
    let arena = TestArena::new();
    let graph = NodeGraph::new(&arena);

    let s0 = arena.push_state(make_state_node(10));
    let s1 = arena.push_state(make_state_node(20));

    let mut outcomes: TinyVec<[AfterStateOutcome; 2]> = TinyVec::new();
    outcomes.push(AfterStateOutcome::new(2, s0));
    outcomes.push(AfterStateOutcome::new(3, s1));
    let after_state_id = arena.push_after_state(AfterState::new(outcomes));

    let before = after_state_outcome_visits(&arena, after_state_id);
    assert_eq!(graph.find_state_with_hash(after_state_id, 20), Some(s1));
    let after = after_state_outcome_visits(&arena, after_state_id);
    assert_eq!(before, after);
}

#[test]
fn get_edge_state_with_hash_can_find_state_through_nested_after_state_children() {
    let arena = TestArena::new();
    let graph = NodeGraph::new(&arena);

    let s0 = arena.push_state(make_state_node(777));

    let mut inner: TinyVec<[AfterStateOutcome; 2]> = TinyVec::new();
    inner.push(AfterStateOutcome::new(5, s0));
    let inner_id = arena.push_after_state(AfterState::new(inner));

    let mut outer: TinyVec<[AfterStateOutcome; 2]> = TinyVec::new();
    outer.push(AfterStateOutcome::new(9, inner_id));
    let outer_id = arena.push_after_state(AfterState::new(outer));

    let edge = PUCTEdge::new();
    edge.set_child(outer_id);

    assert_eq!(graph.get_edge_state_with_hash(&edge, 777), Some(s0));
}

#[test]
fn find_edge_terminal_after_state_terminal_not_first_outcome_returns_terminal() {
    let arena = TestArena::new();
    let graph = NodeGraph::new(&arena);

    let s0 = arena.push_state(make_state_node(1));
    let t0 = make_terminal_node(&arena, 9);

    let mut outcomes: TinyVec<[AfterStateOutcome; 2]> = TinyVec::new();
    outcomes.push(AfterStateOutcome::new(1, s0));
    outcomes.push(AfterStateOutcome::new(2, t0));
    let after_state_id = arena.push_after_state(AfterState::new(outcomes));

    let edge = PUCTEdge::new();
    edge.set_child(after_state_id);

    assert_eq!(graph.find_edge_terminal(&edge), Some(t0));
}

#[test]
fn add_child_to_edge_multiple_times_grows_after_state_outcomes_monotonically() {
    let arena = TestArena::new();
    let graph = NodeGraph::new(&arena);

    let s0 = arena.push_state(make_state_node(100));
    let s1 = arena.push_state(make_state_node(200));
    let s2 = arena.push_state(make_state_node(300));

    let edge = PUCTEdge::new();
    for _ in 0..4 {
        edge.increment_visits();
    }
    assert!(edge.try_set_child(s0));

    graph.add_child_to_edge(&edge, s1);
    let after_state_1 = edge.child().expect("child must exist");
    assert!(after_state_1.is_after_state());
    let out1 = after_state_outcome_visits(&arena, after_state_1);
    assert_eq!(out1.len(), 2);

    // Add a third child; should clone the current afterstate and append one more outcome.
    graph.add_child_to_edge(&edge, s2);
    let after_state_2 = edge.child().expect("child must exist");
    assert!(after_state_2.is_after_state());
    let out2 = after_state_outcome_visits(&arena, after_state_2);
    assert_eq!(out2.len(), 3);

    // The previous afterstate remains unchanged.
    let out1_again = after_state_outcome_visits(&arena, after_state_1);
    assert_eq!(out1, out1_again);
}

#[test]
fn add_child_to_edge_does_not_mutate_edge_visits() {
    let arena = TestArena::new();
    let graph = NodeGraph::new(&arena);

    let s0 = arena.push_state(make_state_node(1));
    let s1 = arena.push_state(make_state_node(2));

    let edge = PUCTEdge::new();
    for _ in 0..9 {
        edge.increment_visits();
    }
    let before = edge.visits();
    assert!(edge.try_set_child(s0));

    graph.add_child_to_edge(&edge, s1);
    assert_eq!(edge.visits(), before);
}

#[test]
fn get_edge_state_with_hash_after_state_three_outcomes_increments_only_matching() {
    let arena = TestArena::new();
    let graph = NodeGraph::new(&arena);

    let s0 = arena.push_state(make_state_node(10));
    let s1 = arena.push_state(make_state_node(20));
    let s2 = arena.push_state(make_state_node(30));

    let mut outcomes: TinyVec<[AfterStateOutcome; 2]> = TinyVec::new();
    outcomes.push(AfterStateOutcome::new(1, s0));
    outcomes.push(AfterStateOutcome::new(2, s1));
    outcomes.push(AfterStateOutcome::new(3, s2));
    let after_state_id = arena.push_after_state(AfterState::new(outcomes));

    let edge = PUCTEdge::new();
    edge.set_child(after_state_id);

    let before = after_state_outcome_visits(&arena, after_state_id);
    assert!(graph.increment_afterstate_visits(&edge, 30));
    assert_eq!(graph.get_edge_state_with_hash(&edge, 30), Some(s2));
    let after = after_state_outcome_visits(&arena, after_state_id);

    assert_eq!(after[0], before[0]);
    assert_eq!(after[1], before[1]);
    assert_eq!(after[2].0, s2);
    assert_eq!(after[2].1, before[2].1 + 1);
}

#[test]
fn get_edge_state_with_hash_after_state_hash_collision_increments_first_match_only() {
    let arena = TestArena::new();
    let graph = NodeGraph::new(&arena);

    // Two distinct state nodes with the same transposition hash (simulates a collision).
    let s0 = arena.push_state(make_state_node(123));
    let s1 = arena.push_state(make_state_node(123));

    let mut outcomes: TinyVec<[AfterStateOutcome; 2]> = TinyVec::new();
    outcomes.push(AfterStateOutcome::new(5, s0));
    outcomes.push(AfterStateOutcome::new(7, s1));
    let after_state_id = arena.push_after_state(AfterState::new(outcomes));

    let edge = PUCTEdge::new();
    edge.set_child(after_state_id);

    let before = after_state_outcome_visits(&arena, after_state_id);
    assert!(graph.increment_afterstate_visits(&edge, 123));
    assert_eq!(graph.get_edge_state_with_hash(&edge, 123), Some(s0));
    let after = after_state_outcome_visits(&arena, after_state_id);

    assert_eq!(after[0].0, s0);
    assert_eq!(after[0].1, before[0].1 + 1);
    assert_eq!(after[1], before[1]);
}

#[test]
fn find_edge_terminal_state_child_returns_none() {
    let arena = TestArena::new();
    let graph = NodeGraph::new(&arena);
    let s0 = arena.push_state(make_state_node(1));

    let edge = PUCTEdge::new();
    assert!(edge.try_set_child(s0));

    assert_eq!(graph.find_edge_terminal(&edge), None);
}

#[test]
fn find_edge_terminal_after_state_multiple_terminal_outcomes_increments_first_terminal_only() {
    let arena = TestArena::new();
    let graph = NodeGraph::new(&arena);

    let t0 = make_terminal_node(&arena, 1);
    let t1 = make_terminal_node(&arena, 2);

    // This AfterState is invalid per is_valid (two terminals), but NodeGraph should still behave
    // deterministically: terminal_outcome() returns the first terminal.
    let mut outcomes: TinyVec<[AfterStateOutcome; 2]> = TinyVec::new();
    outcomes.push(AfterStateOutcome::new(10, t0));
    outcomes.push(AfterStateOutcome::new(20, t1));
    let after_state_id = arena.push_after_state(AfterState::new(outcomes));

    let edge = PUCTEdge::new();
    edge.set_child(after_state_id);

    let before = after_state_outcome_visits(&arena, after_state_id);
    assert!(graph.increment_afterstate_terminal_visits(&edge));
    assert_eq!(graph.find_edge_terminal(&edge), Some(t0));
    let after = after_state_outcome_visits(&arena, after_state_id);

    assert_eq!(after[0].0, t0);
    assert_eq!(after[0].1, before[0].1 + 1);
    assert_eq!(after[1], before[1]);
}

#[test]
#[should_panic]
fn get_edge_state_with_hash_panics_if_after_state_contains_unset_child() {
    let arena = TestArena::new();
    let graph = NodeGraph::new(&arena);

    let mut outcomes: TinyVec<[AfterStateOutcome; 2]> = TinyVec::new();
    outcomes.push(AfterStateOutcome::new(1, NodeId::unset()));
    let after_state_id = arena.push_after_state(AfterState::new(outcomes));

    let edge = PUCTEdge::new();
    edge.set_child(after_state_id);

    // This should panic due to attempting to treat `unset()` as a real node id.
    let _ = graph.get_edge_state_with_hash(&edge, 0);
}

#[test]
fn find_state_with_hash_can_traverse_nested_after_state_nodes() {
    let arena = TestArena::new();
    let graph = NodeGraph::new(&arena);

    let s0 = arena.push_state(make_state_node(777));

    let mut inner: TinyVec<[AfterStateOutcome; 2]> = TinyVec::new();
    inner.push(AfterStateOutcome::new(1, s0));
    let inner_id = arena.push_after_state(AfterState::new(inner));

    let mut outer: TinyVec<[AfterStateOutcome; 2]> = TinyVec::new();
    outer.push(AfterStateOutcome::new(1, inner_id));
    let outer_id = arena.push_after_state(AfterState::new(outer));

    assert_eq!(graph.find_state_with_hash(outer_id, 777), Some(s0));
}

#[test]
#[should_panic]
fn add_child_to_edge_panics_in_debug_if_duplicate_child_is_added_to_after_state() {
    let arena = TestArena::new();
    let graph = NodeGraph::new(&arena);

    let s0 = arena.push_state(make_state_node(1));

    let mut outcomes: TinyVec<[AfterStateOutcome; 2]> = TinyVec::new();
    outcomes.push(AfterStateOutcome::new(2, s0));
    let after_state_id = arena.push_after_state(AfterState::new(outcomes));

    let edge = PUCTEdge::new();
    edge.set_child(after_state_id);

    // Adding the same child again creates a duplicate NodeId in outcomes -> debug_assert should fire.
    graph.add_child_to_edge(&edge, s0);
}

#[test]
#[should_panic]
fn add_child_to_edge_panics_in_debug_if_second_terminal_outcome_is_added() {
    let arena = TestArena::new();
    let graph = NodeGraph::new(&arena);

    let t0 = make_terminal_node(&arena, 1);
    let t1 = make_terminal_node(&arena, 2);

    let mut outcomes: TinyVec<[AfterStateOutcome; 2]> = TinyVec::new();
    outcomes.push(AfterStateOutcome::new(2, t0));
    let after_state_id = arena.push_after_state(AfterState::new(outcomes));

    let edge = PUCTEdge::new();
    edge.set_child(after_state_id);

    // Adding a second terminal makes the AfterState invalid -> debug_assert should fire.
    graph.add_child_to_edge(&edge, t1);
}
