use std::sync::atomic::{AtomicU32, Ordering};

use half::f16;
use model::ActionWithPolicy;
use tinyvec::TinyVec;

use puct::{
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

fn make_node(transposition_hash: u64, priors: &[(u32, f32)]) -> TestStateNode {
    let priors = priors
        .iter()
        .map(|&(a, p)| ActionWithPolicy::new(a, f16::from_f32(p)))
        .collect::<Vec<_>>()
        .into_boxed_slice();

    StateNode::new(transposition_hash, priors, DummyRollup::default())
}

fn make_terminal(arena: &TestArena, value: u32) -> puct::NodeId {
    let r = DummyRollup::default();
    r.set(DummySnapshot(value));
    arena.push_terminal(Terminal::new(r))
}

#[test]
fn harness_writer_traversal_increments_node_and_edge_visits_exactly() {
    // This is a minimal end-to-end harness using the same primitives as PUCT selection:
    // - ensure_frontier_edge
    // - node.increment_visits
    // - edge.increment_visits
    // - add_child_to_edge
    //
    // It does NOT use the PUCT struct (which currently cannot be constructed safely due to
    // self-referential borrowing), but it exercises the same visit-counters.

    let arena = TestArena::new();
    let graph = NodeGraph::new(&arena);

    let root_id = arena.push_state(make_node(1, &[(0, 0.6), (1, 0.4), (2, 0.2)]));
    let root = arena.get_state_node(root_id);

    assert_eq!(root.visits(), 1);
    assert_eq!(root.iter_edges().count(), 0);

    // Step 1: materialize frontier edge and traverse it.
    root.ensure_frontier_edge();
    assert_eq!(root.iter_edges().count(), 1);

    root.increment_visits();
    let (e0, _a0) = root.edge_and_action(0);
    assert_eq!(e0.visits(), 0);
    e0.increment_visits();
    assert_eq!(e0.visits(), 1);
    assert_eq!(root.visits(), 2);

    // Expansion: link a child state.
    let child_id = arena.push_state(make_node(2, &[(0, 1.0)]));
    graph.add_child_to_edge(e0, child_id);

    // Step 2: since the last edge has visits>0, frontier should advance.
    root.ensure_frontier_edge();
    assert_eq!(root.iter_edges().count(), 2);

    root.increment_visits();
    let (e1, _a1) = root.edge_and_action(1);
    assert_eq!(e1.visits(), 0);
    e1.increment_visits();

    assert_eq!(root.visits(), 3);
    assert_eq!(e0.visits(), 1);
    assert_eq!(e1.visits(), 1);

    // Step 3: revisit edge0.
    root.increment_visits();
    e0.increment_visits();

    assert_eq!(root.visits(), 4);
    assert_eq!(e0.visits(), 2);
    assert_eq!(e1.visits(), 1);
}

#[test]
fn harness_after_state_outcome_visits_should_increment_when_selecting_matching_state() {
    // Desired semantics test: when an edge points to an AfterState, selecting a specific outcome
    // (via matching transposition hash) should increment that outcome's visit count.
    //
    // This test is expected to fail until NodeGraph::get_edge_state_with_hash accounts for
    // outcome visits.

    let arena = TestArena::new();
    let graph = NodeGraph::new(&arena);

    let s0 = arena.push_state(make_node(100, &[(0, 1.0)]));
    let s1 = arena.push_state(make_node(200, &[(0, 1.0)]));

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

    assert_eq!(graph.get_edge_state_with_hash(&edge, 200), Some(s1));

    let after = arena
        .get_after_state_node(after_state_id)
        .outcomes
        .iter()
        .map(|o| (o.child(), o.visits()))
        .collect::<Vec<_>>();

    assert_eq!(after[0], before[0]);
    assert_eq!(after[1].0, s1);
    assert_eq!(after[1].1, before[1].1 + 1);
}

#[test]
fn harness_after_state_terminal_outcome_visits_should_increment_when_detecting_terminal() {
    // Desired semantics test: if an edge points to an AfterState with a terminal outcome,
    // repeatedly observing that terminal should increment the terminal outcome's visits.
    //
    // This test is expected to fail until NodeGraph::find_edge_terminal increments
    // terminal outcome visits.

    let arena = TestArena::new();
    let graph = NodeGraph::new(&arena);

    let s0 = arena.push_state(make_node(10, &[(0, 1.0)]));
    let t0 = make_terminal(&arena, 9);

    let mut outcomes: TinyVec<[AfterStateOutcome; 2]> = TinyVec::new();
    outcomes.push(AfterStateOutcome::new(10, s0));
    outcomes.push(AfterStateOutcome::new(20, t0));
    let after_state_id = arena.push_after_state(AfterState::new(outcomes));

    let edge = PUCTEdge::new();
    edge.set_child(after_state_id);

    let before = arena
        .get_after_state_node(after_state_id)
        .outcomes
        .iter()
        .map(|o| (o.child(), o.visits()))
        .collect::<Vec<_>>();

    assert_eq!(graph.find_edge_terminal(&edge), Some(t0));

    let after = arena
        .get_after_state_node(after_state_id)
        .outcomes
        .iter()
        .map(|o| (o.child(), o.visits()))
        .collect::<Vec<_>>();

    // Only terminal outcome increments.
    assert_eq!(after[0], before[0]);
    assert_eq!(after[1].0, t0);
    assert_eq!(after[1].1, before[1].1 + 1);
}

#[test]
fn harness_add_child_to_edge_new_outcome_should_start_at_one() {
    // Desired semantics test: the traversal that discovers/creates a new outcome should
    // credit that outcome with 1 visit.
    //
    // This test is expected to fail until NodeGraph::add_child_to_edge seeds new outcomes at 1.

    let arena = TestArena::new();
    let graph = NodeGraph::new(&arena);

    let existing_child = arena.push_state(make_node(111, &[(0, 1.0)]));

    let edge = PUCTEdge::new();
    edge.increment_visits();
    assert!(edge.try_set_child(existing_child));

    let new_child = arena.push_state(make_node(222, &[(0, 1.0)]));

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
