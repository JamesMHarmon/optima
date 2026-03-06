use std::sync::atomic::{AtomicU32, Ordering};

use crossbeam::channel;
use half::f16;
use model::ActionWithPolicy;

use super::*;
use crate::rollup::{RollupStats, WeightedMerge};

#[derive(Clone, Debug)]
struct DummyState(u64);

impl common::TranspositionHash for DummyState {
    fn transposition_hash(&self) -> u64 {
        self.0
    }
}

struct DummyAnalyzer;

impl model::GameAnalyzer for DummyAnalyzer {
    type Action = ();
    type State = DummyState;
    type Predictions = ();

    fn send(&self, _request_id: model::RequestId, _game_state: &Self::State) {
        unreachable!("send should not be called by next_sim tests")
    }

    fn recv(
        &self,
    ) -> (
        model::RequestId,
        model::GameStateAnalysis<Self::Action, Self::Predictions>,
    ) {
        unreachable!("recv should not be called by next_sim tests")
    }

    fn analyze(
        &self,
        _game_state: &Self::State,
    ) -> model::GameStateAnalysis<Self::Action, Self::Predictions> {
        unreachable!("analyze should not be called by next_sim tests")
    }
}

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

impl From<DummySnapshot> for DummyRollup {
    fn from(value: DummySnapshot) -> Self {
        let out = DummyRollup::default();
        out.set(value);
        out
    }
}

struct DummyValueModel;

impl crate::value_model::ValueModel for DummyValueModel {
    type Predictions = ();
    type Terminal = ();
    type Snapshot = DummySnapshot;
    type Rollup = DummyRollup;

    fn pred_snapshot(&self, _predictions: &Self::Predictions) -> DummySnapshot {
        DummySnapshot(0)
    }

    fn terminal_snapshot(&self, _terminal: &Self::Terminal) -> DummySnapshot {
        DummySnapshot(0)
    }
}

fn terminal_msg(sim_id: usize) -> SimMsg<DummyState, DummySnapshot> {
    SimMsg::Terminal(TerminalSimMsg {
        sim_id,
        path: vec![PathStep {
            node_id: NodeId::from_u32(sim_id as u32),
            edge_index: sim_id,
        }],
        terminal_snapshot: DummySnapshot(0),
    })
}

fn path_sim_id(msg: &SimMsg<DummyState, DummySnapshot>) -> usize {
    let path = match msg {
        SimMsg::Terminal(msg) => &msg.path,
        SimMsg::State(msg) => &msg.path,
        SimMsg::Preempted(msg) => &msg.path,
    };

    path.first().expect("path is not empty").node_id.as_u32() as usize
}

fn make_backprop(
    rx: SimRx<DummyState, DummySnapshot>,
) -> Backpropagator<'static, DummyAnalyzer, DummyValueModel> {
    // Leak these because Backpropagator stores references; tests are short-lived.
    let analyzer = Box::leak(Box::new(DummyAnalyzer));
    let value_model = Box::leak(Box::new(DummyValueModel));
    let store = Box::leak(Box::new(NodeGraphStore::<(), DummyRollup>::new()));

    Backpropagator::new(analyzer, store, value_model, rx)
}

fn one_action_prior() -> Vec<ActionWithPolicy<()>> {
    vec![ActionWithPolicy::new((), f16::from_f32(1.0))]
}

fn make_state_node(store: &PuctStore<DummyAnalyzer, DummyRollup>, hash: u64, snap: u32) -> NodeId {
    let node_id =
        store.create_and_insert_state_node(hash, one_action_prior(), DummySnapshot(snap).into());
    store.state_node(node_id).ensure_frontier_edge();
    node_id
}

fn reserve_path_virtual_loss(store: &PuctStore<DummyAnalyzer, DummyRollup>, path: &[PathStep]) {
    for step in path {
        let node = store.state_node(step.node_id);
        node.increment_virtual_visits();
        node.edge(step.edge_index).increment_virtual_visits();
    }
}

#[test]
fn next_sim_drains_in_order_even_if_msgs_arrive_out_of_order() {
    let (tx, rx) = channel::unbounded::<SimMsg<DummyState, DummySnapshot>>();
    tx.send(terminal_msg(1)).unwrap();
    tx.send(terminal_msg(0)).unwrap();
    tx.send(terminal_msg(2)).unwrap();
    drop(tx);

    let mut backprop = make_backprop(rx);

    let s0 = backprop.next_sim().unwrap();
    let s1 = backprop.next_sim().unwrap();
    let s2 = backprop.next_sim().unwrap();

    assert_eq!(path_sim_id(&s0), 0);
    assert_eq!(path_sim_id(&s1), 1);
    assert_eq!(path_sim_id(&s2), 2);
    assert!(backprop.next_sim().is_none());
}

#[test]
fn next_sim_returns_none_when_channel_closed_and_no_pending() {
    let (_tx, rx) = channel::unbounded::<SimMsg<DummyState, DummySnapshot>>();
    drop(_tx);

    let mut backprop = make_backprop(rx);
    assert!(backprop.next_sim().is_none());
}

#[test]
#[should_panic(expected = "Missing simulation(s): expected sim_id 0")]
fn next_sim_panics_when_expected_sim_id_is_missing_after_channel_closes() {
    let (tx, rx) = channel::unbounded::<SimMsg<DummyState, DummySnapshot>>();
    tx.send(terminal_msg(1)).unwrap();
    drop(tx);

    let mut backprop = make_backprop(rx);
    let _ = backprop.next_sim();
}

#[test]
fn commit_path_increments_node_and_edge_visits_for_direct_state_child() {
    let (_tx, rx) = channel::unbounded::<SimMsg<DummyState, DummySnapshot>>();
    drop(_tx);
    let backprop = make_backprop(rx);

    let store = backprop.store;
    let graph = store.graph();

    let root_id = make_state_node(store, 1, 0);
    let leaf_id = make_state_node(store, 2, 0);

    let root = store.state_node(root_id);
    let edge = root.edge(0);
    graph.add_child_to_edge(edge, leaf_id);

    let path = vec![PathStep {
        node_id: root_id,
        edge_index: 0,
    }];
    reserve_path_virtual_loss(store, &path);

    let node_visits_before = root.visits();
    let edge_visits_before = edge.visits();

    backprop.commit_path(&path, leaf_id);

    assert_eq!(root.visits(), node_visits_before + 1);
    assert_eq!(edge.visits(), edge_visits_before + 1);
    assert_eq!(root.virtual_visits(), 0);
    assert_eq!(edge.virtual_visits(), 0);
}

#[test]
fn commit_path_increments_afterstate_outcome_visits_once_when_edge_is_afterstate() {
    let (_tx, rx) = channel::unbounded::<SimMsg<DummyState, DummySnapshot>>();
    drop(_tx);
    let backprop = make_backprop(rx);

    let store = backprop.store;
    let graph = store.graph();

    let root_id = make_state_node(store, 10, 0);
    let existing_id = make_state_node(store, 11, 0);
    let leaf_id = make_state_node(store, 12, 0);

    let root = store.state_node(root_id);
    let edge = root.edge(0);

    for _ in 0..3 {
        edge.increment_visits();
    }
    graph.add_child_to_edge(edge, existing_id);
    graph.add_child_to_edge(edge, leaf_id);

    assert_eq!(graph.afterstate_outcome_visits(edge, existing_id), Some(3));
    assert_eq!(graph.afterstate_outcome_visits(edge, leaf_id), Some(0));

    let path = vec![PathStep {
        node_id: root_id,
        edge_index: 0,
    }];
    reserve_path_virtual_loss(store, &path);

    let node_visits_before = root.visits();
    let edge_visits_before = edge.visits();

    backprop.commit_path(&path, leaf_id);

    assert_eq!(root.visits(), node_visits_before + 1);
    assert_eq!(edge.visits(), edge_visits_before + 1);
    assert_eq!(graph.afterstate_outcome_visits(edge, existing_id), Some(3));
    assert_eq!(graph.afterstate_outcome_visits(edge, leaf_id), Some(1));
    assert_eq!(root.virtual_visits(), 0);
    assert_eq!(edge.virtual_visits(), 0);
}

#[test]
fn commit_path_increments_outcome_visits_for_internal_and_leaf_edges() {
    let (_tx, rx) = channel::unbounded::<SimMsg<DummyState, DummySnapshot>>();
    drop(_tx);
    let backprop = make_backprop(rx);

    let store = backprop.store;
    let graph = store.graph();

    let root_id = make_state_node(store, 20, 0);
    let mid_id = make_state_node(store, 21, 0);
    let leaf_id = make_state_node(store, 22, 0);

    let alt_root_id = make_state_node(store, 23, 0);
    let alt_mid_id = make_state_node(store, 24, 0);

    let root = store.state_node(root_id);
    let root_edge = root.edge(0);
    for _ in 0..5 {
        root_edge.increment_visits();
    }
    graph.add_child_to_edge(root_edge, alt_root_id);
    graph.add_child_to_edge(root_edge, mid_id);

    let mid = store.state_node(mid_id);
    let mid_edge = mid.edge(0);
    for _ in 0..2 {
        mid_edge.increment_visits();
    }
    graph.add_child_to_edge(mid_edge, alt_mid_id);
    graph.add_child_to_edge(mid_edge, leaf_id);

    assert_eq!(
        graph.afterstate_outcome_visits(root_edge, alt_root_id),
        Some(5)
    );
    assert_eq!(graph.afterstate_outcome_visits(root_edge, mid_id), Some(0));
    assert_eq!(
        graph.afterstate_outcome_visits(mid_edge, alt_mid_id),
        Some(2)
    );
    assert_eq!(graph.afterstate_outcome_visits(mid_edge, leaf_id), Some(0));

    let path = vec![
        PathStep {
            node_id: root_id,
            edge_index: 0,
        },
        PathStep {
            node_id: mid_id,
            edge_index: 0,
        },
    ];
    reserve_path_virtual_loss(store, &path);

    let root_visits_before = root.visits();
    let mid_visits_before = mid.visits();
    let root_edge_visits_before = root_edge.visits();
    let mid_edge_visits_before = mid_edge.visits();

    backprop.commit_path(&path, leaf_id);

    assert_eq!(root.visits(), root_visits_before + 1);
    assert_eq!(mid.visits(), mid_visits_before + 1);
    assert_eq!(root_edge.visits(), root_edge_visits_before + 1);
    assert_eq!(mid_edge.visits(), mid_edge_visits_before + 1);

    assert_eq!(
        graph.afterstate_outcome_visits(root_edge, alt_root_id),
        Some(5)
    );
    assert_eq!(graph.afterstate_outcome_visits(root_edge, mid_id), Some(1));
    assert_eq!(
        graph.afterstate_outcome_visits(mid_edge, alt_mid_id),
        Some(2)
    );
    assert_eq!(graph.afterstate_outcome_visits(mid_edge, leaf_id), Some(1));

    assert_eq!(root.virtual_visits(), 0);
    assert_eq!(root_edge.virtual_visits(), 0);
    assert_eq!(mid.virtual_visits(), 0);
    assert_eq!(mid_edge.virtual_visits(), 0);
}

#[test]
fn terminal_upsert_plus_commit_increments_terminal_outcome_visits_once() {
    let (_tx, rx) = channel::unbounded::<SimMsg<DummyState, DummySnapshot>>();
    drop(_tx);
    let backprop = make_backprop(rx);

    let store = backprop.store;
    let graph = store.graph();

    let root_id = make_state_node(store, 30, 0);
    let existing_id = make_state_node(store, 31, 0);

    let root = store.state_node(root_id);
    let edge = root.edge(0);

    for _ in 0..4 {
        edge.increment_visits();
    }
    graph.add_child_to_edge(edge, existing_id);

    let terminal_id = backprop.upsert_terminal_edge(edge, &DummySnapshot(7));

    assert_eq!(graph.afterstate_outcome_visits(edge, existing_id), Some(4));
    assert_eq!(graph.afterstate_outcome_visits(edge, terminal_id), Some(0));

    let path = vec![PathStep {
        node_id: root_id,
        edge_index: 0,
    }];
    reserve_path_virtual_loss(store, &path);

    backprop.commit_path(&path, terminal_id);

    assert_eq!(graph.afterstate_outcome_visits(edge, existing_id), Some(4));
    assert_eq!(graph.afterstate_outcome_visits(edge, terminal_id), Some(1));
}
