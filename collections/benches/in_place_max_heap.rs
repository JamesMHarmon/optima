use std::cmp::Ordering;

use criterion::{
    BatchSize, BenchmarkId, Criterion, SamplingMode, black_box, criterion_group, criterion_main,
};
use half::f16;
use model::ActionWithPolicy;

use collections::{Comparator, InPlaceMaxHeap};

criterion_group!(benches, bench_in_place_max_heap);
criterion_main!(benches);

const SIZES: &[usize] = &[2048];
const SIZES_2: &[usize] = &[32];

fn bench_in_place_max_heap(c: &mut Criterion) {
    let mut group = c.benchmark_group("in_place_max_heap");
    group.sample_size(50);
    group.sampling_mode(SamplingMode::Flat);

    for &n in SIZES {
        group.bench_with_input(BenchmarkId::new("first_extract_u32", n), &n, |b, &n| {
            b.iter_batched(
                || {
                    let items = make_u32_items(n, 0xC0FFEE);
                    InPlaceMaxHeap::new(items)
                },
                |heap| {
                    // First extraction pays heapify cost (lazy heapify).
                    heap.extract_next_to_end();
                    black_box(heap.remaining());
                },
                BatchSize::SmallInput,
            )
        });

        for &k in SIZES_2 {
            let k = k.min(n);
            group.bench_with_input(
                BenchmarkId::new("extract_u32_after_lazy_init", format!("n={n}/k={k}")),
                &(n, k),
                |b, &(n, k)| {
                    b.iter_batched(
                        || {
                            let items = make_u32_items(n, 0xBADC0DE);
                            let heap = InPlaceMaxHeap::new(items);
                            // Trigger lazy heapify (and 1 extraction) during setup.
                            heap.extract_next_to_end();
                            heap
                        },
                        |heap| {
                            // Do remaining extracts (k includes the setup extract).
                            for _ in 1..k {
                                heap.extract_next_to_end();
                            }
                            black_box(heap.remaining());
                        },
                        BatchSize::SmallInput,
                    )
                },
            );
        }

        group.bench_with_input(
            BenchmarkId::new("materialize_get_u32", format!("n={n}/k={}", SIZES_2[0])),
            &n,
            |b, &n| {
                let k = SIZES_2[0].min(n);
                b.iter_batched(
                    || {
                        let items = make_u32_items(n, 0xA11CE);

                        InPlaceMaxHeap::new(items)
                    },
                    |heap| {
                        for i in 0..k {
                            while heap.total_len() - heap.remaining() < i + 1 {
                                heap.extract_next_to_end();
                            }
                            black_box(*heap.extracted(i));
                        }
                    },
                    BatchSize::SmallInput,
                )
            },
        );

        group.bench_with_input(
            BenchmarkId::new("materialize_iter_u32", format!("n={n}/k={}", SIZES_2[0])),
            &n,
            |b, &n| {
                let k = SIZES_2[0].min(n);
                b.iter_batched(
                    || {
                        let items = make_u32_items(n, 0xB16B00B5);

                        InPlaceMaxHeap::new(items)
                    },
                    |heap| {
                        for i in 0..k {
                            heap.extract_next_to_end();
                            black_box(*heap.extracted(i));
                        }
                    },
                    BatchSize::SmallInput,
                )
            },
        );

        group.bench_with_input(
            BenchmarkId::new("first_extract_action_with_policy", n),
            &n,
            |b, &n| {
                b.iter_batched(
                    || {
                        let items = make_action_with_policy_items(n, 0xDEADBEEF).into_boxed_slice();
                        InPlaceMaxHeap::with_comparator(items, PolicyScoreCmp)
                    },
                    |heap| {
                        heap.extract_next_to_end();
                        black_box(heap.remaining());
                    },
                    BatchSize::SmallInput,
                )
            },
        );

        // Extraction benchmark for ActionWithPolicy, after heapify.
        let k = (n / 10).max(1).min(n);
        group.bench_with_input(
            BenchmarkId::new(
                "extract_action_with_policy_after_lazy_init",
                format!("n={n}/k={k}"),
            ),
            &(n, k),
            |b, &(n, k)| {
                b.iter_batched(
                    || {
                        let items = make_action_with_policy_items(n, 0xFEEDFACE).into_boxed_slice();
                        let heap = InPlaceMaxHeap::with_comparator(items, PolicyScoreCmp);
                        heap.extract_next_to_end();
                        heap
                    },
                    |heap| {
                        for _ in 1..k {
                            heap.extract_next_to_end();
                        }
                        black_box(heap.remaining());
                    },
                    BatchSize::SmallInput,
                )
            },
        );

        group.bench_with_input(
            BenchmarkId::new(
                "materialize_get_action_with_policy",
                format!("n={n}/k={}", SIZES_2[0]),
            ),
            &n,
            |b, &n| {
                let k = SIZES_2[0].min(n);
                b.iter_batched(
                    || {
                        let items = make_action_with_policy_items(n, 0x12345678).into_boxed_slice();

                        InPlaceMaxHeap::with_comparator(items, PolicyScoreCmp)
                    },
                    |heap| {
                        for i in 0..k {
                            while heap.total_len() - heap.remaining() < i + 1 {
                                heap.extract_next_to_end();
                            }
                            let awp = heap.extracted(i);
                            black_box(awp.policy_score().to_f32());
                        }
                    },
                    BatchSize::SmallInput,
                )
            },
        );
    }

    group.finish();
}

#[derive(Clone, Copy, Default)]
struct PolicyScoreCmp;

impl<A> Comparator<ActionWithPolicy<A>> for PolicyScoreCmp {
    #[inline]
    fn cmp(&self, a: &ActionWithPolicy<A>, b: &ActionWithPolicy<A>) -> Ordering {
        let pa = a.policy_score().to_f32();
        let pb = b.policy_score().to_f32();
        pa.partial_cmp(&pb).unwrap_or(Ordering::Equal)
    }
}

fn make_u32_items(len: usize, seed: u64) -> Box<[u32]> {
    let mut x = seed;
    let mut v = Vec::with_capacity(len);
    for _ in 0..len {
        x = lcg_step(x);
        v.push((x >> 32) as u32);
    }
    v.into_boxed_slice()
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
    // Same constants as PCG/LCG-style mixes; good enough for deterministic bench data.
    x.wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407)
}
