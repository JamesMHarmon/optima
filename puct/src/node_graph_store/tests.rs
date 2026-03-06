use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::thread;

use half::f16;
use model::ActionWithPolicy;

use super::*;
use crate::rollup::{RollupStats, WeightedMerge};

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
    fn from(s: DummySnapshot) -> Self {
        let r = DummyRollup::default();
        r.set(s);
        r
    }
}

type TestStore = NodeGraphStore<u32, DummyRollup>;

fn one_prior() -> Vec<ActionWithPolicy<u32>> {
    vec![ActionWithPolicy::new(0u32, f16::from_f32(1.0))]
}

fn make_store_with_node(hash: u64) -> (TestStore, NodeId) {
    let store = TestStore::new();
    let node_id = store.create_and_insert_state_node(hash, one_prior(), DummySnapshot(0).into());
    store.state_node(node_id).ensure_frontier_edge();
    (store, node_id)
}

/// When the transposition table has no entry for the hash, returns Claimed
/// (genuinely new leaf that needs expansion).
#[test]
fn returns_claimed_when_hash_not_in_transposition_table() {
    let store = TestStore::new();
    let edge = PUCTEdge::new();

    assert!(matches!(
        store.link_or_try_claim(&edge, 0xdeadbeef),
        LeafResult::Claimed
    ));
}

/// When the hash is in the transposition table and the edge is unlinked,
/// links the edge and returns Known(node_id).
#[test]
fn links_edge_and_returns_known_when_hash_known_and_edge_unlinked() {
    let (store, node_id) = make_store_with_node(42);
    let edge = PUCTEdge::new();

    let result = store.link_or_try_claim(&edge, 42);

    assert!(matches!(result, LeafResult::Known(id) if id == node_id));
    assert_eq!(edge.child(), Some(node_id));
}

/// Regression test for the original CAS bug:
/// When the hash is in the transposition table but the edge was ALREADY linked
/// by a racing thread (try_set_child CAS would fail), the function must still
/// return Known(node_id) rather than Claimed or Preempted.
///
/// The old code was: `edge.try_set_child(*existing_id).ok().map(|_| *existing_id)`
/// which turned the Err (CAS lost the race) into None, making the sim think it
/// had found a new leaf and sending a spurious SimMsg::State.
#[test]
fn returns_known_even_when_edge_already_linked_by_racing_thread() {
    let (store, node_id) = make_store_with_node(42);
    let edge = PUCTEdge::new();

    // Simulate a racing thread that already won the CAS and linked the edge.
    edge.set_child(node_id);

    // Despite the CAS that try_set_child would perform internally failing,
    // the result must be Known — not Claimed or Preempted.
    let result = store.link_or_try_claim(&edge, 42);

    assert!(matches!(result, LeafResult::Known(id) if id == node_id));
}

/// One thread gets Claimed, all others get Preempted — never an incorrect
/// Known or a second Claimed.
#[test]
fn concurrent_calls_one_claims_rest_are_preempted() {
    let store = Arc::new(TestStore::new());
    // Don't insert into transposition_table — all callers see a genuinely new hash.
    let edge = Arc::new(PUCTEdge::new());
    const THREADS: usize = 64;

    let handles: Vec<_> = (0..THREADS)
        .map(|_| {
            let store = Arc::clone(&store);
            let edge = Arc::clone(&edge);
            thread::spawn(move || store.link_or_try_claim(&edge, 99))
        })
        .collect();

    let mut claimed = 0usize;
    let mut preempted = 0usize;
    for handle in handles {
        match handle.join().unwrap() {
            LeafResult::Claimed => claimed += 1,
            LeafResult::Preempted => preempted += 1,
            LeafResult::Known(_) => panic!("unexpected Known — hash was never inserted"),
        }
    }

    assert_eq!(claimed, 1, "exactly one thread must win the claim");
    assert_eq!(
        preempted,
        THREADS - 1,
        "all other threads must be Preempted"
    );
}
