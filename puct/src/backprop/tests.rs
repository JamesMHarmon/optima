use std::sync::atomic::{AtomicU32, Ordering};

use crossbeam::channel;

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

    fn analyze(&self, _request_id: model::RequestId, _game_state: &Self::State) {
        unreachable!("analyze should not be called by next_sim tests")
    }

    fn recv(
        &self,
    ) -> (
        model::RequestId,
        model::GameStateAnalysis<Self::Action, Self::Predictions>,
    ) {
        unreachable!("recv should not be called by next_sim tests")
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
    type State = DummyState;
    type Predictions = ();
    type Terminal = ();
    type Rollup = DummyRollup;

    fn pred_snapshot(&self, _state: &Self::State, _predictions: &Self::Predictions) -> DummySnapshot {
        DummySnapshot(0)
    }

    fn terminal_snapshot(
        &self,
        _state: &Self::State,
        _terminal: &Self::Terminal,
    ) -> DummySnapshot {
        DummySnapshot(0)
    }
}

#[derive(Clone, Default)]
struct DummyExpansions;

impl crate::analysis_coordinator::InFlightExpansions for DummyExpansions {
    type State = DummyState;

    fn analyze(&self, _request_id: model::RequestId, _game_state: Self::State) {
        unreachable!("analyze should not be called by next_sim tests")
    }

    fn complete(&self, _hash: u64) {
        unreachable!("complete should not be called by next_sim tests")
    }
}

fn terminal_msg(sim_id: usize) -> SimMsg<DummyState> {
    SimMsg::Terminal {
        sim_id,
        path: vec![PathStep {
            node_id: NodeId::from_u32(sim_id as u32),
            edge_index: sim_id,
        }],
    }
}

fn path_sim_id(pending: &PendingSim<DummyState>) -> usize {
    pending
        .path()
        .first()
        .expect("path is not empty")
        .node_id
        .as_u32() as usize
}

fn make_backprop(rx: SimRx<DummyState>) -> Backpropagator<'static, DummyAnalyzer, DummyValueModel, DummyExpansions> {
    // Leak these because Backpropagator stores references; tests are short-lived.
    let analyzer: &'static DummyAnalyzer = Box::leak(Box::new(DummyAnalyzer));
    let value_model: &'static DummyValueModel = Box::leak(Box::new(DummyValueModel));
    let store: &'static PuctStore<DummyAnalyzer, DummyRollup> =
        Box::leak(Box::new(NodeGraphStore::<(), DummyRollup>::new()));

    Backpropagator::new(analyzer, store, value_model, DummyExpansions, rx)
}

#[test]
fn next_sim_drains_in_order_even_if_msgs_arrive_out_of_order() {
    let (tx, rx) = channel::unbounded::<SimMsg<DummyState>>();
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
    let (_tx, rx) = channel::unbounded::<SimMsg<DummyState>>();
    drop(_tx);

    let mut backprop = make_backprop(rx);
    assert!(backprop.next_sim().is_none());
}

#[test]
#[should_panic(expected = "Missing simulation(s): expected sim_id 0")]
fn next_sim_panics_when_expected_sim_id_is_missing_after_channel_closes() {
    let (tx, rx) = channel::unbounded::<SimMsg<DummyState>>();
    tx.send(terminal_msg(1)).unwrap();
    drop(tx);

    let mut backprop = make_backprop(rx);
    let _ = backprop.next_sim();
}
