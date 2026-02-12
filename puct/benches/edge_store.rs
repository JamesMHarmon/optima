use criterion::{BatchSize, Criterion, black_box, criterion_group, criterion_main};
use half::f16;
use model::ActionWithPolicy;

use puct::StateNode;
use puct::{RollupStats, WeightedMerge};

use std::sync::atomic::{AtomicU32, Ordering};

criterion_group!(benches, bench_edge_store);
criterion_main!(benches);

const EDGE_COUNT: usize = 500;

type BenchNode = StateNode<u32, DummyRollup, ()>;

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

fn bench_edge_store(c: &mut Criterion) {
    let mut group = c.benchmark_group("edge_store");

    group.bench_function("edges_iter_500", |b| {
        b.iter_batched(
            || make_node_with_edges(EDGE_COUNT),
            |node: BenchNode| {
                for (edge, awp) in node.iter_edges_with_policy() {
                    black_box(edge);
                    black_box(awp);
                }
            },
            BatchSize::SmallInput,
        )
    });

    group.finish();
}

fn make_node_with_edges(edge_count: usize) -> BenchNode {
    let priors = make_action_with_policy_items(edge_count, 0xC0FFEE).into_boxed_slice();
    let node = StateNode::new(0, priors, (), DummyRollup::default());

    let mut i = 0;
    while i < edge_count {
        node.ensure_frontier_edge();

        // Mark the newly-materialized frontier edge as visited so the frontier can advance.
        let (edge, _action) = node.edge_and_action(i);
        edge.increment_visits();

        i += 1;
    }

    node
}

fn make_action_with_policy_items(len: usize, seed: u64) -> Vec<ActionWithPolicy<u32>> {
    let mut x = seed;
    let mut v = Vec::with_capacity(len);

    for action in 0..(len as u32) {
        x = lcg_step(x);
        // Deterministic pseudo-random score in (0, 1).
        let u = ((x >> 11) as u32) as f32 / (u32::MAX as f32);
        let score = f16::from_f32(u);
        v.push(ActionWithPolicy::new(action, score));
    }

    v
}

#[inline]
fn lcg_step(x: u64) -> u64 {
    x.wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407)
}
