use std::sync::atomic::{AtomicU32, Ordering};

use half::f16;
use model::ActionWithPolicy;
use tinyvec::TinyVec;

use crate::{
    AfterState, AfterStateOutcome, EdgeInfo, NodeArena, RollupStats, Terminal, WeightedMerge,
};

use super::StateNode;

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

fn make_node(priors: &[(u32, f32)]) -> TestStateNode {
    let priors = priors
        .iter()
        .map(|&(a, p)| ActionWithPolicy::new(a, f16::from_f32(p)))
        .collect::<Vec<_>>()
        .into_boxed_slice();

    StateNode::new(0, priors, DummyRollup::default())
}

fn edge_count(node: &TestStateNode) -> usize {
    node.iter_edges().count()
}

fn frontier_edge_count(node: &TestStateNode) -> usize {
    node.iter_edges().filter(|e| e.visits() == 0).count()
}

#[test]
fn ensure_frontier_edge_materializes_at_most_one_unvisited_edge() {
    let node = make_node(&[(0, 0.1), (1, 0.2), (2, 0.3)]);

    assert_eq!(edge_count(&node), 0);

    node.ensure_frontier_edge();
    assert_eq!(edge_count(&node), 1);
    assert_eq!(frontier_edge_count(&node), 1);

    // Repeated calls should not create additional unvisited edges.
    for _ in 0..10 {
        node.ensure_frontier_edge();
        assert_eq!(edge_count(&node), 1);
        assert_eq!(frontier_edge_count(&node), 1);
    }
}

#[test]
fn frontier_advances_only_after_visiting_frontier_edge() {
    let node = make_node(&[(0, 0.2), (1, 0.5), (2, 0.4)]);

    node.ensure_frontier_edge();
    assert_eq!(edge_count(&node), 1);

    // Visit the frontier edge.
    let (e0, _a0) = node.edge_and_action(0);
    assert_eq!(e0.visits(), 0);
    e0.increment_visits();

    // Now the frontier can advance, materializing a new edge.
    node.ensure_frontier_edge();
    assert_eq!(edge_count(&node), 2);
    assert_eq!(frontier_edge_count(&node), 1);

    // Repeated ensures still keep only one unvisited edge.
    node.ensure_frontier_edge();
    assert_eq!(edge_count(&node), 2);
    assert_eq!(frontier_edge_count(&node), 1);
}

#[test]
fn edge_and_action_and_iterators_are_aligned() {
    let node = make_node(&[(10, 0.9), (11, 0.8), (12, 0.7), (13, 0.6)]);

    // Materialize 4 edges; visit first 3 so we end with exactly one frontier edge.
    for i in 0..4 {
        node.ensure_frontier_edge();
        let (edge, _action) = node.edge_and_action(i);
        if i < 3 {
            edge.increment_visits();
        }
    }

    assert_eq!(edge_count(&node), 4);
    assert_eq!(frontier_edge_count(&node), 1);

    let edges_only: Vec<*const _> = node.iter_edges().map(|e| e as *const _).collect();
    let edges_with_policy: Vec<(*const _, u32)> = node
        .iter_edges_with_policy()
        .map(|(e, awp)| (e as *const _, *awp.action()))
        .collect();

    assert_eq!(edges_only.len(), edges_with_policy.len());

    for i in 0..edges_only.len() {
        let (edge_i, action_i) = node.edge_and_action(i);

        assert_eq!(edges_only[i], edge_i as *const _);
        assert_eq!(edges_with_policy[i].0, edge_i as *const _);
        assert_eq!(edges_with_policy[i].1, *action_i);
    }
}

#[test]
fn iter_edge_info_matches_edges_with_policy() {
    let node = make_node(&[(0, 0.25), (1, 0.75), (2, 0.50)]);

    // Materialize all edges.
    for i in 0..3 {
        node.ensure_frontier_edge();
        let (edge, _action) = node.edge_and_action(i);
        edge.increment_visits();
    }

    let nodes: TestArena = NodeArena::new();

    let with_policy: Vec<(*const _, u32, f32, u32)> = node
        .iter_edges_with_policy()
        .map(|(e, awp)| {
            (
                e as *const _,
                *awp.action(),
                awp.policy_score().to_f32(),
                e.visits(),
            )
        })
        .collect();

    let infos: Vec<EdgeInfo<'_, u32, DummySnapshot>> = node.iter_edge_info(&nodes).collect();

    assert_eq!(with_policy.len(), infos.len());

    for (i, info) in infos.iter().enumerate() {
        assert_eq!(info.edge_index, i);
        assert_eq!(*info.action, with_policy[i].1);
        assert_eq!(info.policy_prior, with_policy[i].2);
        assert_eq!(info.visits, with_policy[i].3);
        assert!(info.snapshot.is_none());
    }

    let snapshots: Vec<(*const _, DummySnapshot)> = node
        .iter_edge_rollups(&nodes)
        .map(|(e, s)| (e as *const _, s))
        .collect();

    assert_eq!(snapshots.len(), 0);
}

#[test]
fn materialization_follows_policy_prior_order() {
    // Intentionally unsorted priors.
    let node = make_node(&[(0, 0.10), (1, 0.90), (2, 0.40), (3, 0.70)]);

    // Materialize all 4 edges.
    for i in 0..4 {
        node.ensure_frontier_edge();
        let (edge, _action) = node.edge_and_action(i);
        edge.increment_visits();
    }

    let actions_in_edge_order: Vec<u32> = node
        .iter_edges_with_policy()
        .map(|(_edge, awp)| *awp.action())
        .collect();

    assert_eq!(actions_in_edge_order, vec![1, 3, 2, 0]);
}

#[test]
fn frontier_materializes_next_best_policy_prior_step_by_step() {
    // Intentionally unsorted priors with distinct values.
    let priors = vec![(0, 0.10), (1, 0.90), (2, 0.40), (3, 0.70), (4, 0.20)];
    let node = make_node(&priors);

    let mut expected = priors.clone();
    expected.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let expected_actions: Vec<u32> = expected.into_iter().map(|(a, _)| a).collect();

    for i in 0..expected_actions.len() {
        node.ensure_frontier_edge();
        assert_eq!(edge_count(&node), i + 1);
        assert_eq!(frontier_edge_count(&node), 1);

        // Repeated calls should not create additional frontier edges.
        node.ensure_frontier_edge();
        assert_eq!(edge_count(&node), i + 1);
        assert_eq!(frontier_edge_count(&node), 1);

        let (edge, action) = node.edge_and_action(i);
        assert_eq!(*action, expected_actions[i]);

        // Visit all but the last edge so that the frontier can advance.
        if i + 1 < expected_actions.len() {
            edge.increment_visits();
        }
    }

    // After materializing all edges, the last one is still the only frontier edge.
    assert_eq!(edge_count(&node), expected_actions.len());
    assert_eq!(frontier_edge_count(&node), 1);
}

#[test]
fn edge_snapshots_return_child_rollup_for_state_and_terminal_children() {
    let nodes: TestArena = NodeArena::new();

    let empty_priors: Box<[ActionWithPolicy<u32>]> = Vec::new().into_boxed_slice();
    let state_id = nodes.push_state(StateNode::new(1, empty_priors, DummyRollup::default()));
    nodes
        .get_state_node(state_id)
        .rollup_stats()
        .set(DummySnapshot(5));

    let terminal_rollup = {
        let r = DummyRollup::default();
        r.set(DummySnapshot(7));
        r
    };
    let terminal_id = nodes.push_terminal(Terminal::new(terminal_rollup));

    // State child
    let node_state = make_node(&[(0, 0.5)]);
    node_state.ensure_frontier_edge();
    let (edge0, _action0) = node_state.edge_and_action(0);
    edge0.set_child(state_id);

    let snaps: Vec<DummySnapshot> = node_state
        .iter_edge_rollups(&nodes)
        .map(|(_e, s)| s)
        .collect();

    assert_eq!(snaps, vec![DummySnapshot(5)]);

    // Terminal child
    let node_term = make_node(&[(0, 0.5)]);
    node_term.ensure_frontier_edge();
    let (edge1, _action1) = node_term.edge_and_action(0);
    edge1.set_child(terminal_id);

    let infos: Vec<EdgeInfo<'_, u32, DummySnapshot>> = node_term.iter_edge_info(&nodes).collect();
    assert_eq!(infos.len(), 1);
    assert_eq!(infos[0].snapshot, Some(DummySnapshot(7)));
}

#[test]
fn edge_snapshots_aggregate_after_state_outcomes_weighted() {
    let nodes: TestArena = NodeArena::new();

    let empty_priors: Box<[ActionWithPolicy<u32>]> = Vec::new().into_boxed_slice();
    let state_id = nodes.push_state(StateNode::new(1, empty_priors, DummyRollup::default()));
    nodes
        .get_state_node(state_id)
        .rollup_stats()
        .set(DummySnapshot(5));

    let terminal_rollup = {
        let r = DummyRollup::default();
        r.set(DummySnapshot(7));
        r
    };
    let terminal_id = nodes.push_terminal(Terminal::new(terminal_rollup));

    // AfterState with two outcomes: 2 visits to state(5), 3 visits to terminal(7)
    let mut outcomes: TinyVec<[AfterStateOutcome; 2]> = TinyVec::new();
    outcomes.push(AfterStateOutcome::new(2, state_id));
    outcomes.push(AfterStateOutcome::new(3, terminal_id));
    let after_state_id = nodes.push_after_state(AfterState::new(outcomes));

    let node = make_node(&[(0, 0.5)]);
    node.ensure_frontier_edge();
    let (edge, _action) = node.edge_and_action(0);
    edge.set_child(after_state_id);

    let snap = node
        .iter_edge_rollups(&nodes)
        .next()
        .map(|(_e, s)| s)
        .expect("expected snapshot for afterstate child");

    assert_eq!(snap, DummySnapshot(5 * 2 + 7 * 3));
}
