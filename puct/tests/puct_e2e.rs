use std::sync::atomic::{AtomicU32, Ordering};

use half::f16;
use model::ActionWithPolicy;
use tinyvec::TinyVec;

use puct::{
    AfterState, AfterStateOutcome, NodeArena, NodeGraph, NodeId, PUCTEdge, RollupStats, StateNode,
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

fn make_node(transposition_hash: u64, prior_value: u32, priors: &[(u32, f32)]) -> TestStateNode {
    let priors = priors
        .iter()
        .map(|&(a, p)| ActionWithPolicy::new(a, f16::from_f32(p)))
        .collect::<Vec<_>>()
        .into_boxed_slice();

    let r = DummyRollup::default();
    r.set(DummySnapshot(prior_value));

    StateNode::new(transposition_hash, priors, r)
}

fn make_terminal(arena: &TestArena, value: u32) -> NodeId {
    let r = DummyRollup::default();
    r.set(DummySnapshot(value));
    arena.push_terminal(Terminal::new(r))
}

fn bump_visits(edge: &PUCTEdge, times: u32) {
    for _ in 0..times {
        edge.increment_visits();
    }
}

#[test]
fn e2e_rollup_recompute_weights_children_by_edge_visits_and_includes_prior() {
    let arena = TestArena::new();
    let graph = NodeGraph::new(&arena);

    let root_id = arena.push_state(make_node(1, 5, &[(0, 0.8), (1, 0.2)]));
    let root = arena.get_state_node(root_id);

    root.ensure_frontier_edge();
    let (edge0, _) = root.edge_and_action(0);
    // `ensure_frontier_edge` only materializes a new edge once the previous frontier edge has
    // been visited.
    edge0.increment_visits();
    root.ensure_frontier_edge();
    let (edge1, _) = root.edge_and_action(1);

    let t0 = make_terminal(&arena, 2);
    graph.add_child_to_edge(edge0, t0);

    let leaf_state_id = arena.push_state(make_node(2, 7, &[(0, 1.0)]));
    let leaf_terminal_id = make_terminal(&arena, 11);

    let mut outcomes: TinyVec<[AfterStateOutcome; 2]> = TinyVec::new();
    outcomes.push(AfterStateOutcome::new(1, leaf_state_id));
    outcomes.push(AfterStateOutcome::new(4, leaf_terminal_id));
    let after_state_id = arena.push_after_state(AfterState::new(outcomes));
    edge1.set_child(after_state_id);

    bump_visits(edge0, 2); // already has 1 visit; total becomes 3 (contributes 2*3)
    bump_visits(edge1, 2); // contributes after_state_snapshot*2

    root.recompute_rollup(&arena);

    // after_state_snapshot = 7*1 + 11*4 = 51
    // root_rollup = prior(5)*1 + edge0(2)*3 + edge1(51)*2 = 5 + 6 + 102 = 113
    assert_eq!(root.rollup_stats().snapshot(), DummySnapshot(113));
}

#[test]
fn e2e_rollup_recompute_ignores_unvisited_edges() {
    let arena = TestArena::new();
    let graph = NodeGraph::new(&arena);

    let root_id = arena.push_state(make_node(1, 9, &[(0, 1.0), (1, 1.0)]));
    let root = arena.get_state_node(root_id);

    root.ensure_frontier_edge();
    let (edge0, _) = root.edge_and_action(0);
    edge0.increment_visits();
    root.ensure_frontier_edge();
    let (edge1, _) = root.edge_and_action(1);

    let t0 = make_terminal(&arena, 100);
    let t1 = make_terminal(&arena, 200);
    graph.add_child_to_edge(edge0, t0);
    graph.add_child_to_edge(edge1, t1);

    bump_visits(edge0, 1); // already has 1 visit; total becomes 2
    // edge1 stays at 0 visits -> should not contribute

    root.recompute_rollup(&arena);

    // prior 9 + (100*2) = 209
    assert_eq!(root.rollup_stats().snapshot(), DummySnapshot(209));
}

#[test]
fn e2e_terminal_merge_accumulates_when_terminal_already_reachable_via_edge() {
    let arena = TestArena::new();
    let graph = NodeGraph::new(&arena);

    let terminal_id = make_terminal(&arena, 10);

    let edge = PUCTEdge::new();
    edge.set_child(terminal_id);

    let snapshot_to_add = DummySnapshot(7);

    let found_terminal_id = graph
        .find_edge_terminal(&edge)
        .expect("terminal must be reachable");
    assert_eq!(found_terminal_id, terminal_id);

    let terminal_node = arena.get_terminal_node(found_terminal_id);
    assert_eq!(terminal_node.rollup_stats().snapshot(), DummySnapshot(10));

    terminal_node.rollup_stats().accumulate(&snapshot_to_add);
    assert_eq!(terminal_node.rollup_stats().snapshot(), DummySnapshot(17));
}

#[test]
fn e2e_add_child_to_edge_converts_to_afterstate_and_preserves_existing_visits() {
    // This validates current behavior for transposition linking:
    // - first child sets edge.child directly
    // - second distinct child converts edge.child into an AfterState
    // - the existing child becomes an outcome with visits == edge.visits()
    // - the new outcome starts at 1 visit (credit this traversal)

    let arena = TestArena::new();
    let graph = NodeGraph::new(&arena);

    let s0 = arena.push_state(make_node(10, 0, &[(0, 1.0)]));
    let s1 = arena.push_state(make_node(20, 0, &[(0, 1.0)]));

    let edge = PUCTEdge::new();
    bump_visits(&edge, 4);

    graph.add_child_to_edge(&edge, s0);

    bump_visits(&edge, 3); // edge.visits now 7; should be captured into s0 outcome during conversion
    graph.add_child_to_edge(&edge, s1);

    let after_state_id = edge.child().expect("edge should have a child");
    assert!(after_state_id.is_after_state());

    let after_state = arena.get_after_state_node(after_state_id);

    let o0 = after_state
        .outcomes
        .iter()
        .find(|o| o.child() == s0)
        .expect("existing child outcome must exist");
    let o1 = after_state
        .outcomes
        .iter()
        .find(|o| o.child() == s1)
        .expect("new child outcome must exist");

    assert_eq!(o0.visits(), edge.visits());
    assert_eq!(o1.visits(), 1);
}

#[test]
fn e2e_get_edge_state_with_hash_matches_state_outcomes_in_afterstate() {
    let arena = TestArena::new();
    let graph = NodeGraph::new(&arena);

    let s0 = arena.push_state(make_node(111, 1, &[(0, 1.0)]));
    let s1 = arena.push_state(make_node(222, 2, &[(0, 1.0)]));
    let t0 = make_terminal(&arena, 9);

    let mut outcomes: TinyVec<[AfterStateOutcome; 2]> = TinyVec::new();
    outcomes.push(AfterStateOutcome::new(1, s0));
    outcomes.push(AfterStateOutcome::new(1, s1));
    outcomes.push(AfterStateOutcome::new(1, t0));
    let after_state_id = arena.push_after_state(AfterState::new(outcomes));

    debug_assert!(arena.get_after_state_node(after_state_id).is_valid());

    let edge = PUCTEdge::new();
    edge.set_child(after_state_id);

    assert_eq!(graph.get_edge_state_with_hash(&edge, 111), Some(s0));
    assert_eq!(graph.get_edge_state_with_hash(&edge, 222), Some(s1));
    assert_eq!(graph.get_edge_state_with_hash(&edge, 333), None);
}

#[test]
fn e2e_find_edge_terminal_finds_terminal_through_afterstate() {
    let arena = TestArena::new();
    let graph = NodeGraph::new(&arena);

    let s0 = arena.push_state(make_node(1, 0, &[(0, 1.0)]));
    let t0 = make_terminal(&arena, 42);

    let mut outcomes: TinyVec<[AfterStateOutcome; 2]> = TinyVec::new();
    outcomes.push(AfterStateOutcome::new(1, s0));
    outcomes.push(AfterStateOutcome::new(1, t0));
    let after_state_id = arena.push_after_state(AfterState::new(outcomes));

    let edge = PUCTEdge::new();
    edge.set_child(after_state_id);

    assert_eq!(graph.find_edge_terminal(&edge), Some(t0));
}
