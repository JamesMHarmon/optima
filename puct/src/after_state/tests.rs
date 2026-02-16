use std::sync::atomic::{AtomicU32, Ordering};

use half::f16;
use model::ActionWithPolicy;
use tinyvec::TinyVec;

use crate::after_state::{AfterState, AfterStateOutcome};
use crate::node::StateNode;
use crate::node_arena::{NodeArena, NodeId};
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

fn make_state_node(hash: u64, value: u32) -> TestStateNode {
    let priors = [(0u32, 1.0f32)]
        .into_iter()
        .map(|(a, p)| ActionWithPolicy::new(a, f16::from_f32(p)))
        .collect::<Vec<_>>()
        .into_boxed_slice();

    let node = StateNode::new(hash, priors, DummyRollup::default());
    node.rollup_stats().set(DummySnapshot(value));
    node
}

#[test]
fn after_state_outcome_increment_visits_is_exact() {
    let outcome = AfterStateOutcome::new(0, NodeId::from_u32(0b11 << 30));

    assert_eq!(outcome.visits(), 0);
    outcome.increment_visits();
    assert_eq!(outcome.visits(), 1);
    outcome.increment_visits();
    outcome.increment_visits();
    assert_eq!(outcome.visits(), 3);
}

#[test]
fn after_state_is_valid_enforces_invariants() {
    let arena = TestArena::new();

    let s0 = arena.push_state(make_state_node(1, 10));
    let s1 = arena.push_state(make_state_node(2, 20));

    let t = {
        let r = DummyRollup::default();
        r.set(DummySnapshot(99));
        arena.push_terminal(Terminal::new(r))
    };

    let mut ok: TinyVec<[AfterStateOutcome; 2]> = TinyVec::new();
    ok.push(AfterStateOutcome::new(1, s0));
    ok.push(AfterStateOutcome::new(2, t));
    let after_state_ok = AfterState::new(ok);
    assert!(after_state_ok.is_valid());

    let mut dup: TinyVec<[AfterStateOutcome; 2]> = TinyVec::new();
    dup.push(AfterStateOutcome::new(1, s1));
    dup.push(AfterStateOutcome::new(2, s1));
    let after_state_dup = AfterState::new(dup);
    assert!(!after_state_dup.is_valid());

    let mut two_terms: TinyVec<[AfterStateOutcome; 2]> = TinyVec::new();
    two_terms.push(AfterStateOutcome::new(1, t));
    two_terms.push(AfterStateOutcome::new(2, t));
    let after_state_two_terms = AfterState::new(two_terms);
    assert!(!after_state_two_terms.is_valid());
}

#[test]
fn after_state_snapshot_weights_by_outcome_visits() {
    let arena = TestArena::new();

    let s0 = arena.push_state(make_state_node(10, 5));
    let s1 = arena.push_state(make_state_node(20, 7));

    let mut outcomes: TinyVec<[AfterStateOutcome; 2]> = TinyVec::new();
    outcomes.push(AfterStateOutcome::new(2, s0));
    outcomes.push(AfterStateOutcome::new(3, s1));

    let after_state = AfterState::new(outcomes);
    let snap = after_state.snapshot(&arena);
    assert_eq!(snap, DummySnapshot(5 * 2 + 7 * 3));

    // Increment one outcome and ensure the snapshot changes accordingly.
    after_state.outcomes[0].increment_visits();
    let snap2 = after_state.snapshot(&arena);
    assert_eq!(snap2, DummySnapshot(5 * 3 + 7 * 3));
}

#[test]
fn after_state_snapshot_ignores_zero_visit_outcomes() {
    let arena = TestArena::new();

    let s0 = arena.push_state(make_state_node(10, 5));
    let s1 = arena.push_state(make_state_node(20, 7));

    let mut outcomes: TinyVec<[AfterStateOutcome; 2]> = TinyVec::new();
    outcomes.push(AfterStateOutcome::new(0, s0));
    outcomes.push(AfterStateOutcome::new(3, s1));

    let after_state = AfterState::new(outcomes);
    let snap = after_state.snapshot(&arena);
    assert_eq!(snap, DummySnapshot(7 * 3));
}

#[test]
fn after_state_snapshot_all_zero_visits_is_zero() {
    let arena = TestArena::new();

    let s0 = arena.push_state(make_state_node(10, 5));
    let s1 = arena.push_state(make_state_node(20, 7));

    let mut outcomes: TinyVec<[AfterStateOutcome; 2]> = TinyVec::new();
    outcomes.push(AfterStateOutcome::new(0, s0));
    outcomes.push(AfterStateOutcome::new(0, s1));

    let after_state = AfterState::new(outcomes);
    let snap = after_state.snapshot(&arena);
    assert_eq!(snap, DummySnapshot(0));
}

#[test]
#[should_panic]
fn after_state_snapshot_panics_if_outcome_child_is_after_state() {
    let arena = TestArena::new();

    let s0 = arena.push_state(make_state_node(1, 10));

    let mut inner: TinyVec<[AfterStateOutcome; 2]> = TinyVec::new();
    inner.push(AfterStateOutcome::new(1, s0));
    let inner_id = arena.push_after_state(AfterState::new(inner));

    let mut outer: TinyVec<[AfterStateOutcome; 2]> = TinyVec::new();
    outer.push(AfterStateOutcome::new(1, inner_id));
    let outer_after_state = AfterState::new(outer);

    // `iter_outcomes` explicitly panics for AfterState children.
    let _ = outer_after_state.snapshot(&arena);
}

#[test]
#[should_panic]
fn after_state_snapshot_panics_if_outcome_child_is_unset() {
    let arena = TestArena::new();

    let mut outcomes: TinyVec<[AfterStateOutcome; 2]> = TinyVec::new();
    outcomes.push(AfterStateOutcome::new(1, NodeId::unset()));
    let after_state = AfterState::new(outcomes);

    // This should panic in debug builds (and would be invalid for traversal anyway).
    let _ = after_state.snapshot(&arena);
}

#[test]
fn terminal_outcome_is_independent_of_visits() {
    let arena = TestArena::new();

    let s0 = arena.push_state(make_state_node(1, 10));
    let t0 = {
        let r = DummyRollup::default();
        r.set(DummySnapshot(1));
        arena.push_terminal(Terminal::new(r))
    };

    let mut outcomes: TinyVec<[AfterStateOutcome; 2]> = TinyVec::new();
    outcomes.push(AfterStateOutcome::new(100, s0));
    outcomes.push(AfterStateOutcome::new(0, t0));
    let after_state = AfterState::new(outcomes);

    assert_eq!(after_state.terminal_outcome().unwrap().child(), t0);
    after_state.outcomes[1].increment_visits();
    assert_eq!(after_state.terminal_outcome().unwrap().child(), t0);
}

#[test]
fn after_state_outcome_clone_preserves_child_and_visits() {
    let child = NodeId::from_u32(0b11 << 30);
    let outcome = AfterStateOutcome::new(123, child);
    let cloned = outcome.clone();

    assert_eq!(cloned.child(), outcome.child());
    assert_eq!(cloned.visits(), outcome.visits());
}

#[test]
fn after_state_outcome_as_tuple_matches_accessors() {
    let child = NodeId::from_u32(0b11 << 30);
    let outcome = AfterStateOutcome::new(7, child);
    assert_eq!((outcome.child(), outcome.visits()), (child, 7));
}

#[test]
fn terminal_outcome_returns_none_when_no_terminal_child() {
    let arena = TestArena::new();

    let s0 = arena.push_state(make_state_node(1, 10));
    let s1 = arena.push_state(make_state_node(2, 20));

    let mut outcomes: TinyVec<[AfterStateOutcome; 2]> = TinyVec::new();
    outcomes.push(AfterStateOutcome::new(1, s0));
    outcomes.push(AfterStateOutcome::new(2, s1));
    let after_state = AfterState::new(outcomes);

    assert!(after_state.terminal_outcome().is_none());
}

#[test]
fn terminal_outcome_returns_first_terminal_when_multiple_present() {
    let arena = TestArena::new();

    let t0 = {
        let r = DummyRollup::default();
        r.set(DummySnapshot(1));
        arena.push_terminal(Terminal::new(r))
    };
    let t1 = {
        let r = DummyRollup::default();
        r.set(DummySnapshot(2));
        arena.push_terminal(Terminal::new(r))
    };

    // Invalid per is_valid, but terminal_outcome() should be deterministic.
    let mut outcomes: TinyVec<[AfterStateOutcome; 2]> = TinyVec::new();
    outcomes.push(AfterStateOutcome::new(1, t0));
    outcomes.push(AfterStateOutcome::new(2, t1));
    let after_state = AfterState::new(outcomes);

    assert_eq!(after_state.terminal_outcome().unwrap().child(), t0);
}

#[test]
fn after_state_is_valid_rejects_after_state_children() {
    let arena = TestArena::new();

    let s0 = arena.push_state(make_state_node(1, 10));
    let mut inner: TinyVec<[AfterStateOutcome; 2]> = TinyVec::new();
    inner.push(AfterStateOutcome::new(1, s0));
    let inner_id = arena.push_after_state(AfterState::new(inner));

    let mut outcomes: TinyVec<[AfterStateOutcome; 2]> = TinyVec::new();
    outcomes.push(AfterStateOutcome::new(1, inner_id));
    let after_state = AfterState::new(outcomes);

    assert!(!after_state.is_valid());
}

#[test]
fn after_state_is_valid_rejects_unset_children() {
    // This is a stricter expectation than the current implementation enforces.
    // Leaving it as a test to catch regressions once `is_valid` is tightened.
    let mut outcomes: TinyVec<[AfterStateOutcome; 2]> = TinyVec::new();
    outcomes.push(AfterStateOutcome::new(1, NodeId::unset()));
    let after_state = AfterState::new(outcomes);
    assert!(!after_state.is_valid());
}
