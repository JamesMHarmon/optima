use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use append_only_vec::AppendOnlyVec;
use arc_swap::ArcSwap;
use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use parking_lot::{Mutex as ParkingMutex, RwLock as ParkingRwLock};

use rcu_append_buffer::RcuAppendBuffer;

criterion_group!(
    benches,
    bench_one_writer_readers,
    bench_read_only_many_readers
);
criterion_main!(benches);

const TARGET_LENS: &[usize] = &[128];
const READER_COUNTS: &[usize] = &[2];
const WRITE_TO_READ_RATIOS: &[usize] = &[1000];

const READ_ONLY_READER_COUNTS: &[usize] = &[1, 2, 4, 8];
const READ_ONLY_READ_OPS_PER_READER: usize = 50_000;

#[derive(Clone, Copy, Debug)]
struct Entry {
    a: u32,
    b: u32,
    c: u32,
}

impl Entry {
    fn from_seed(seed: u32) -> Self {
        Self {
            a: seed,
            b: seed.wrapping_add(1),
            c: seed.wrapping_add(2),
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum ImplKind {
    RcuAppendBuffer,
    AppendOnlyVec,
    ArcSwapVec,
    ParkingMutexVec,
    ParkingRwLockVec,
    StdRwLockVec,
}

impl ImplKind {
    fn name(self) -> &'static str {
        match self {
            ImplKind::RcuAppendBuffer => "rcu_append_buffer",
            ImplKind::AppendOnlyVec => "append_only_vec",
            ImplKind::ArcSwapVec => "arc_swap_vec",
            ImplKind::ParkingMutexVec => "parking_lot_mutex_vec",
            ImplKind::ParkingRwLockVec => "parking_lot_rwlock_vec",
            ImplKind::StdRwLockVec => "std_rwlock_vec",
        }
    }
}

fn bench_one_writer_readers(c: &mut Criterion) {
    if std::env::var_os("OPTIMA_LATENCY").is_some() {
        run_latency_reports();
        return;
    }

    let mut group = c.benchmark_group("one_writer_readers");
    group.measurement_time(Duration::from_secs(3));

    for &target_len in TARGET_LENS {
        for &readers in READER_COUNTS {
            for &ratio in WRITE_TO_READ_RATIOS {
                // Each "read op" is a full scan over all values currently in the collection.
                // The writer starts at len=0, pushes 1 value, then for each subsequent push
                // readers perform ~`ratio` scans per reader (concurrently) until len reaches `target_len`.
                // Throughput is reported as total entries visited (sum of scan lengths).
                // Before the kth push (k starts at 2), the length is (k-1), so we scan
                // lengths 1..=(target_len-1), `ratio * readers` times each.
                let scans_per_step = ratio.saturating_mul(readers).max(1) as u64;
                let n = target_len.saturating_sub(1) as u64;
                let expected_total_entries_visited = scans_per_step
                    .saturating_mul(n)
                    .saturating_mul(n.saturating_add(1))
                    / 2;
                group.throughput(Throughput::Elements(expected_total_entries_visited.max(1)));

                for &impl_kind in &[
                    ImplKind::RcuAppendBuffer,
                    ImplKind::AppendOnlyVec,
                    ImplKind::ArcSwapVec,
                    ImplKind::ParkingMutexVec,
                    ImplKind::ParkingRwLockVec,
                    ImplKind::StdRwLockVec,
                ] {
                    let id = BenchmarkId::new(
                        impl_kind.name(),
                        format!("len={target_len}/readers={readers}/w2r=1:{ratio}"),
                    );

                    group.bench_with_input(id, &(target_len, readers, ratio), |b, input| {
                        let (target_len, readers, ratio) = *input;
                        b.iter_custom(|_iters| {
                            // We run the scenario `iters` times; returning elapsed time keeps Criterion happy.
                            let start = Instant::now();
                            for _ in 0.._iters {
                                run_scenario(impl_kind, target_len, readers, ratio);
                            }
                            start.elapsed()
                        })
                    });
                }
            }
        }
    }

    group.finish();
}

fn bench_read_only_many_readers(c: &mut Criterion) {
    if std::env::var_os("OPTIMA_LATENCY").is_some() {
        return;
    }

    let mut group = c.benchmark_group("read_only_many_readers");
    group.measurement_time(Duration::from_secs(3));

    for &target_len in TARGET_LENS {
        for &readers in READ_ONLY_READER_COUNTS {
            let expected_total_entries_visited = (target_len as u64)
                .saturating_mul(readers as u64)
                .saturating_mul(READ_ONLY_READ_OPS_PER_READER as u64)
                .max(1);
            group.throughput(Throughput::Elements(expected_total_entries_visited));

            for &impl_kind in &[
                ImplKind::RcuAppendBuffer,
                ImplKind::AppendOnlyVec,
                ImplKind::ArcSwapVec,
                ImplKind::ParkingMutexVec,
                ImplKind::ParkingRwLockVec,
                ImplKind::StdRwLockVec,
            ] {
                let id = BenchmarkId::new(
                    impl_kind.name(),
                    format!("len={target_len}/readers={readers}"),
                );

                group.bench_with_input(id, &(target_len, readers), |b, input| {
                    let (target_len, readers) = *input;
                    b.iter_custom(|iters| {
                        let start = Instant::now();
                        for _ in 0..iters {
                            run_read_only_scenario(impl_kind, target_len, readers);
                        }
                        start.elapsed()
                    })
                });
            }
        }
    }

    group.finish();
}

fn run_read_only_scenario(impl_kind: ImplKind, target_len: usize, readers: usize) {
    match impl_kind {
        ImplKind::RcuAppendBuffer => {
            let buf: Arc<RcuAppendBuffer<Entry>> = Arc::new(RcuAppendBuffer::new());
            for i in 0..target_len as u32 {
                buf.push(Entry::from_seed(i));
            }

            let start_barrier = Arc::new(std::sync::Barrier::new(readers));
            std::thread::scope(|s| {
                for _ in 0..readers {
                    let buf = buf.clone();
                    let start_barrier = start_barrier.clone();
                    s.spawn(move || {
                        start_barrier.wait();
                        for _ in 0..READ_ONLY_READ_OPS_PER_READER {
                            let snap = buf.snapshot();
                            consume_all(snap.as_slice());
                        }
                    });
                }
            });
        }
        ImplKind::AppendOnlyVec => {
            let buf: Arc<AppendOnlyVec<Entry>> = Arc::new(AppendOnlyVec::new());
            for i in 0..target_len as u32 {
                buf.push(Entry::from_seed(i));
            }

            let start_barrier = Arc::new(std::sync::Barrier::new(readers));
            std::thread::scope(|s| {
                for _ in 0..readers {
                    let buf = buf.clone();
                    let start_barrier = start_barrier.clone();
                    s.spawn(move || {
                        start_barrier.wait();
                        for _ in 0..READ_ONLY_READ_OPS_PER_READER {
                            consume_all_append_only_vec(&buf);
                        }
                    });
                }
            });
        }
        ImplKind::ArcSwapVec => {
            let initial: Vec<Entry> = (0..target_len as u32).map(Entry::from_seed).collect();
            let buf: Arc<ArcSwap<Vec<Entry>>> = Arc::new(ArcSwap::from_pointee(initial));

            let start_barrier = Arc::new(std::sync::Barrier::new(readers));
            std::thread::scope(|s| {
                for _ in 0..readers {
                    let buf = buf.clone();
                    let start_barrier = start_barrier.clone();
                    s.spawn(move || {
                        start_barrier.wait();
                        for _ in 0..READ_ONLY_READ_OPS_PER_READER {
                            let snap = buf.load_full();
                            consume_all(&snap);
                        }
                    });
                }
            });
        }
        ImplKind::ParkingMutexVec => {
            let initial: Vec<Entry> = (0..target_len as u32).map(Entry::from_seed).collect();
            let buf: Arc<ParkingMutex<Vec<Entry>>> = Arc::new(ParkingMutex::new(initial));

            let start_barrier = Arc::new(std::sync::Barrier::new(readers));
            std::thread::scope(|s| {
                for _ in 0..readers {
                    let buf = buf.clone();
                    let start_barrier = start_barrier.clone();
                    s.spawn(move || {
                        start_barrier.wait();
                        for _ in 0..READ_ONLY_READ_OPS_PER_READER {
                            let g = buf.lock();
                            consume_all(&g);
                        }
                    });
                }
            });
        }
        ImplKind::ParkingRwLockVec => {
            let initial: Vec<Entry> = (0..target_len as u32).map(Entry::from_seed).collect();
            let buf: Arc<ParkingRwLock<Vec<Entry>>> = Arc::new(ParkingRwLock::new(initial));

            let start_barrier = Arc::new(std::sync::Barrier::new(readers));
            std::thread::scope(|s| {
                for _ in 0..readers {
                    let buf = buf.clone();
                    let start_barrier = start_barrier.clone();
                    s.spawn(move || {
                        start_barrier.wait();
                        for _ in 0..READ_ONLY_READ_OPS_PER_READER {
                            let g = buf.read();
                            consume_all(&g);
                        }
                    });
                }
            });
        }
        ImplKind::StdRwLockVec => {
            let initial: Vec<Entry> = (0..target_len as u32).map(Entry::from_seed).collect();
            let buf: Arc<std::sync::RwLock<Vec<Entry>>> = Arc::new(std::sync::RwLock::new(initial));

            let start_barrier = Arc::new(std::sync::Barrier::new(readers));
            std::thread::scope(|s| {
                for _ in 0..readers {
                    let buf = buf.clone();
                    let start_barrier = start_barrier.clone();
                    s.spawn(move || {
                        start_barrier.wait();
                        for _ in 0..READ_ONLY_READ_OPS_PER_READER {
                            let g = buf.read().unwrap();
                            consume_all(&g);
                        }
                    });
                }
            });
        }
    }
}

#[derive(Default)]
struct LatencyResult {
    publish_ns: Vec<u64>,
    reader_ns: Vec<u64>,
}

#[derive(Clone, Copy)]
struct LatencySummary {
    count: usize,
    mean_ns: u64,
    p50_ns: u64,
    p95_ns: u64,
    p99_ns: u64,
    max_ns: u64,
}

fn summarize_ns(mut values: Vec<u64>) -> Option<LatencySummary> {
    if values.is_empty() {
        return None;
    }
    values.sort_unstable();
    let count = values.len();
    let sum: u128 = values.iter().map(|v| *v as u128).sum();
    let mean_ns = (sum / count as u128) as u64;
    let p50_ns = values[count.saturating_mul(50) / 100];
    let p95_ns = values[count.saturating_mul(95) / 100];
    let p99_ns = values[count.saturating_mul(99) / 100];
    let max_ns = *values.last().unwrap();

    Some(LatencySummary {
        count,
        mean_ns,
        p50_ns,
        p95_ns,
        p99_ns,
        max_ns,
    })
}

fn ns_to_us(ns: u64) -> f64 {
    ns as f64 / 1_000.0
}

fn print_latency(
    kind: ImplKind,
    target_len: usize,
    readers: usize,
    ratio: usize,
    result: LatencyResult,
) {
    let publish = summarize_ns(result.publish_ns);
    let reader = summarize_ns(result.reader_ns);

    println!(
        "\nlatency {} len={} readers={} w2r=1:{}",
        kind.name(),
        target_len,
        readers,
        ratio
    );

    if let Some(s) = publish {
        println!(
            "  publish (ns): n={} mean={:.2}us p50={:.2}us p95={:.2}us p99={:.2}us max={:.2}us",
            s.count,
            ns_to_us(s.mean_ns),
            ns_to_us(s.p50_ns),
            ns_to_us(s.p95_ns),
            ns_to_us(s.p99_ns),
            ns_to_us(s.max_ns)
        );
    } else {
        println!("  publish: n=0");
    }

    if let Some(s) = reader {
        println!(
            "  reader  (ns): n={} mean={:.2}us p50={:.2}us p95={:.2}us p99={:.2}us max={:.2}us",
            s.count,
            ns_to_us(s.mean_ns),
            ns_to_us(s.p50_ns),
            ns_to_us(s.p95_ns),
            ns_to_us(s.p99_ns),
            ns_to_us(s.max_ns)
        );
    } else {
        println!("  reader: n=0");
    }
}

fn run_latency_reports() {
    // Run each scenario once and print latency summaries.
    // This mode is enabled by setting OPTIMA_LATENCY=1.
    for &target_len in TARGET_LENS {
        for &readers in READER_COUNTS {
            for &ratio in WRITE_TO_READ_RATIOS {
                for &impl_kind in &[
                    ImplKind::RcuAppendBuffer,
                    ImplKind::AppendOnlyVec,
                    ImplKind::ArcSwapVec,
                    ImplKind::ParkingMutexVec,
                    ImplKind::ParkingRwLockVec,
                    ImplKind::StdRwLockVec,
                ] {
                    let result = run_scenario_latency(impl_kind, target_len, readers, ratio);
                    print_latency(impl_kind, target_len, readers, ratio, result);
                }
            }
        }
    }
}

fn run_scenario_latency(
    impl_kind: ImplKind,
    target_len: usize,
    readers: usize,
    ratio: usize,
) -> LatencyResult {
    match impl_kind {
        ImplKind::RcuAppendBuffer => run_rcu_append_buffer_latency(readers, ratio, target_len),
        ImplKind::AppendOnlyVec => run_append_only_vec_latency(readers, ratio, target_len),
        ImplKind::ArcSwapVec => run_arc_swap_vec_latency(readers, ratio, target_len),
        ImplKind::ParkingMutexVec => run_parking_mutex_vec_latency(readers, ratio, target_len),
        ImplKind::ParkingRwLockVec => run_parking_rwlock_vec_latency(readers, ratio, target_len),
        ImplKind::StdRwLockVec => run_std_rwlock_vec_latency(readers, ratio, target_len),
    }
}

fn run_scenario(impl_kind: ImplKind, target_len: usize, readers: usize, ratio: usize) {
    run_impl(impl_kind, readers, ratio, target_len)
}

fn run_impl(impl_kind: ImplKind, readers: usize, ratio: usize, target_len: usize) {
    match impl_kind {
        ImplKind::RcuAppendBuffer => run_rcu_append_buffer(readers, ratio, target_len),
        ImplKind::AppendOnlyVec => run_append_only_vec(readers, ratio, target_len),
        ImplKind::ArcSwapVec => run_arc_swap_vec(readers, ratio, target_len),
        ImplKind::ParkingMutexVec => run_parking_mutex_vec(readers, ratio, target_len),
        ImplKind::ParkingRwLockVec => run_parking_rwlock_vec(readers, ratio, target_len),
        ImplKind::StdRwLockVec => run_std_rwlock_vec(readers, ratio, target_len),
    }
}

fn consume_all(values: &[Entry]) {
    // Requirement: each read iterates over the entire contents of the collection.
    let mut acc = 0u64;
    for v in values {
        acc = acc.wrapping_add(v.a as u64);
        acc = acc.wrapping_add(v.b as u64);
        acc = acc.wrapping_add(v.c as u64);
    }
    black_box(acc);
}

fn consume_all_append_only_vec(values: &AppendOnlyVec<Entry>) {
    // Requirement: each read iterates over the entire contents of the collection.
    // AppendOnlyVec is safe to index concurrently with appends.
    let n = values.len();
    let mut acc = 0u64;
    for i in 0..n {
        let v = &values[i];
        acc = acc.wrapping_add(v.a as u64);
        acc = acc.wrapping_add(v.b as u64);
        acc = acc.wrapping_add(v.c as u64);
    }
    black_box(acc);
}

fn run_append_only_vec(readers: usize, ratio: usize, target_len: usize) {
    if target_len == 0 {
        return;
    }

    let buf: Arc<AppendOnlyVec<Entry>> = Arc::new(AppendOnlyVec::new());

    let start_barrier = Arc::new(std::sync::Barrier::new(readers + 1));
    let started = Arc::new(AtomicBool::new(false));
    let done = Arc::new(AtomicBool::new(false));
    let read_ops = Arc::new(AtomicUsize::new(0));
    let writer_seed = Arc::new(AtomicU32::new(0));

    std::thread::scope(|s| {
        {
            let buf = buf.clone();
            let start_barrier = start_barrier.clone();
            let started = started.clone();
            let done = done.clone();
            let read_ops = read_ops.clone();
            let writer_seed = writer_seed.clone();

            s.spawn(move || {
                start_barrier.wait();

                buf.push(Entry::from_seed(
                    writer_seed.fetch_add(1, Ordering::Relaxed),
                ));
                started.store(true, Ordering::Release);

                let reads_per_write = ratio.saturating_mul(readers).max(1);
                let mut next_threshold = reads_per_write;
                let mut pushed = 1usize;

                while pushed < target_len {
                    while read_ops.load(Ordering::Acquire) < next_threshold {
                        std::hint::spin_loop();
                    }

                    buf.push(Entry::from_seed(
                        writer_seed.fetch_add(1, Ordering::Relaxed),
                    ));
                    pushed += 1;
                    next_threshold = next_threshold.saturating_add(reads_per_write);
                }

                done.store(true, Ordering::Release);
            });
        }

        for _ in 0..readers {
            let buf = buf.clone();
            let start_barrier = start_barrier.clone();
            let started = started.clone();
            let done = done.clone();
            let read_ops = read_ops.clone();

            s.spawn(move || {
                start_barrier.wait();
                while !started.load(Ordering::Acquire) {
                    std::hint::spin_loop();
                }

                while !done.load(Ordering::Acquire) {
                    consume_all_append_only_vec(&buf);
                    read_ops.fetch_add(1, Ordering::Release);
                }
            });
        }
    });
}

fn run_append_only_vec_latency(readers: usize, ratio: usize, target_len: usize) -> LatencyResult {
    if target_len == 0 {
        return LatencyResult::default();
    }

    let buf: Arc<AppendOnlyVec<Entry>> = Arc::new(AppendOnlyVec::new());

    let start_barrier = Arc::new(std::sync::Barrier::new(readers + 1));
    let started = Arc::new(AtomicBool::new(false));
    let done = Arc::new(AtomicBool::new(false));
    let read_ops = Arc::new(AtomicUsize::new(0));
    let writer_seed = Arc::new(AtomicU32::new(0));

    std::thread::scope(|s| {
        let writer_handle = {
            let buf = buf.clone();
            let start_barrier = start_barrier.clone();
            let started = started.clone();
            let done = done.clone();
            let read_ops = read_ops.clone();
            let writer_seed = writer_seed.clone();

            s.spawn(move || {
                let mut publish_ns: Vec<u64> = Vec::with_capacity(target_len.max(1));
                start_barrier.wait();

                let t0 = Instant::now();
                buf.push(Entry::from_seed(
                    writer_seed.fetch_add(1, Ordering::Relaxed),
                ));
                publish_ns.push(t0.elapsed().as_nanos() as u64);
                started.store(true, Ordering::Release);

                let reads_per_write = ratio.saturating_mul(readers).max(1);
                let mut next_threshold = reads_per_write;
                let mut pushed = 1usize;

                while pushed < target_len {
                    while read_ops.load(Ordering::Acquire) < next_threshold {
                        std::hint::spin_loop();
                    }

                    let t0 = Instant::now();
                    buf.push(Entry::from_seed(
                        writer_seed.fetch_add(1, Ordering::Relaxed),
                    ));
                    publish_ns.push(t0.elapsed().as_nanos() as u64);
                    pushed += 1;
                    next_threshold = next_threshold.saturating_add(reads_per_write);
                }

                done.store(true, Ordering::Release);
                publish_ns
            })
        };

        let mut reader_handles = Vec::with_capacity(readers);
        for _ in 0..readers {
            let buf = buf.clone();
            let start_barrier = start_barrier.clone();
            let started = started.clone();
            let done = done.clone();
            let read_ops = read_ops.clone();

            reader_handles.push(s.spawn(move || {
                let mut reader_ns: Vec<u64> = Vec::new();
                start_barrier.wait();
                while !started.load(Ordering::Acquire) {
                    std::hint::spin_loop();
                }

                while !done.load(Ordering::Acquire) {
                    let t0 = Instant::now();
                    consume_all_append_only_vec(&buf);
                    reader_ns.push(t0.elapsed().as_nanos() as u64);
                    read_ops.fetch_add(1, Ordering::Release);
                }

                reader_ns
            }));
        }

        let publish_ns = writer_handle.join().unwrap();
        let mut reader_ns = Vec::new();
        for h in reader_handles {
            reader_ns.extend(h.join().unwrap());
        }

        LatencyResult {
            publish_ns,
            reader_ns,
        }
    })
}

fn run_rcu_append_buffer(readers: usize, ratio: usize, target_len: usize) {
    let buf: Arc<RcuAppendBuffer<Entry>> = Arc::new(RcuAppendBuffer::new());

    let start_barrier = Arc::new(std::sync::Barrier::new(readers + 1));
    let started = Arc::new(AtomicBool::new(false));
    let done = Arc::new(AtomicBool::new(false));
    let read_ops = Arc::new(AtomicUsize::new(0));
    let writer_seed = Arc::new(AtomicU32::new(0));

    std::thread::scope(|s| {
        {
            let buf = buf.clone();
            let start_barrier = start_barrier.clone();
            let started = started.clone();
            let done = done.clone();
            let read_ops = read_ops.clone();
            let writer_seed = writer_seed.clone();

            s.spawn(move || {
                start_barrier.wait();

                // Requirement: length starts at 0, then a single item is pushed.
                buf.push(Entry::from_seed(
                    writer_seed.fetch_add(1, Ordering::Relaxed),
                ));
                started.store(true, Ordering::Release);

                let reads_per_write = ratio.saturating_mul(readers).max(1);
                let mut next_threshold = reads_per_write;

                while buf.len() < target_len {
                    while read_ops.load(Ordering::Acquire) < next_threshold {
                        std::hint::spin_loop();
                    }

                    buf.push(Entry::from_seed(
                        writer_seed.fetch_add(1, Ordering::Relaxed),
                    ));
                    next_threshold = next_threshold.saturating_add(reads_per_write);
                }

                done.store(true, Ordering::Release);
            });
        }

        for _ in 0..readers {
            let buf = buf.clone();
            let start_barrier = start_barrier.clone();
            let started = started.clone();
            let done = done.clone();
            let read_ops = read_ops.clone();

            s.spawn(move || {
                start_barrier.wait();
                while !started.load(Ordering::Acquire) {
                    std::hint::spin_loop();
                }

                while !done.load(Ordering::Acquire) {
                    let snap = buf.snapshot();
                    consume_all(snap.as_slice());
                    read_ops.fetch_add(1, Ordering::Release);
                }
            });
        }
    });
}

fn run_rcu_append_buffer_latency(readers: usize, ratio: usize, target_len: usize) -> LatencyResult {
    let buf: Arc<RcuAppendBuffer<Entry>> = Arc::new(RcuAppendBuffer::new());

    let start_barrier = Arc::new(std::sync::Barrier::new(readers + 1));
    let started = Arc::new(AtomicBool::new(false));
    let done = Arc::new(AtomicBool::new(false));
    let read_ops = Arc::new(AtomicUsize::new(0));
    let writer_seed = Arc::new(AtomicU32::new(0));

    std::thread::scope(|s| {
        let writer_handle = {
            let buf = buf.clone();
            let start_barrier = start_barrier.clone();
            let started = started.clone();
            let done = done.clone();
            let read_ops = read_ops.clone();
            let writer_seed = writer_seed.clone();

            s.spawn(move || {
                let mut publish_ns: Vec<u64> = Vec::with_capacity(target_len.max(1));
                start_barrier.wait();

                let t0 = Instant::now();
                buf.push(Entry::from_seed(
                    writer_seed.fetch_add(1, Ordering::Relaxed),
                ));
                publish_ns.push(t0.elapsed().as_nanos() as u64);
                started.store(true, Ordering::Release);

                let reads_per_write = ratio.saturating_mul(readers).max(1);
                let mut next_threshold = reads_per_write;

                while buf.len() < target_len {
                    while read_ops.load(Ordering::Acquire) < next_threshold {
                        std::hint::spin_loop();
                    }

                    let t0 = Instant::now();
                    buf.push(Entry::from_seed(
                        writer_seed.fetch_add(1, Ordering::Relaxed),
                    ));
                    publish_ns.push(t0.elapsed().as_nanos() as u64);

                    next_threshold = next_threshold.saturating_add(reads_per_write);
                }

                done.store(true, Ordering::Release);
                publish_ns
            })
        };

        let mut reader_handles = Vec::with_capacity(readers);
        for _ in 0..readers {
            let buf = buf.clone();
            let start_barrier = start_barrier.clone();
            let started = started.clone();
            let done = done.clone();
            let read_ops = read_ops.clone();

            reader_handles.push(s.spawn(move || {
                let mut reader_ns: Vec<u64> = Vec::new();
                start_barrier.wait();
                while !started.load(Ordering::Acquire) {
                    std::hint::spin_loop();
                }

                while !done.load(Ordering::Acquire) {
                    let t0 = Instant::now();
                    let snap = buf.snapshot();
                    consume_all(snap.as_slice());
                    reader_ns.push(t0.elapsed().as_nanos() as u64);
                    read_ops.fetch_add(1, Ordering::Release);
                }

                reader_ns
            }));
        }

        let publish_ns = writer_handle.join().unwrap();
        let mut reader_ns = Vec::new();
        for h in reader_handles {
            reader_ns.extend(h.join().unwrap());
        }

        LatencyResult {
            publish_ns,
            reader_ns,
        }
    })
}

fn run_arc_swap_vec(readers: usize, ratio: usize, target_len: usize) {
    // ArcSwap baseline: copy-on-write Vec (writer clones Vec, appends, then swaps).
    let buf: Arc<ArcSwap<Vec<Entry>>> = Arc::new(ArcSwap::from_pointee(Vec::new()));

    let start_barrier = Arc::new(std::sync::Barrier::new(readers + 1));
    let started = Arc::new(AtomicBool::new(false));
    let done = Arc::new(AtomicBool::new(false));
    let read_ops = Arc::new(AtomicUsize::new(0));
    let writer_seed = Arc::new(AtomicU32::new(0));

    std::thread::scope(|s| {
        {
            let buf = buf.clone();
            let start_barrier = start_barrier.clone();
            let started = started.clone();
            let done = done.clone();
            let read_ops = read_ops.clone();
            let writer_seed = writer_seed.clone();

            s.spawn(move || {
                start_barrier.wait();

                // Requirement: length starts at 0, then a single item is pushed.
                {
                    let mut next = (*buf.load_full()).clone();
                    next.push(Entry::from_seed(
                        writer_seed.fetch_add(1, Ordering::Relaxed),
                    ));
                    buf.store(Arc::new(next));
                }
                started.store(true, Ordering::Release);

                let reads_per_write = ratio.saturating_mul(readers).max(1);
                let mut next_threshold = reads_per_write;

                loop {
                    let len = buf.load_full().len();
                    if len >= target_len {
                        break;
                    }

                    while read_ops.load(Ordering::Acquire) < next_threshold {
                        std::hint::spin_loop();
                    }

                    let mut next = (*buf.load_full()).clone();
                    next.push(Entry::from_seed(
                        writer_seed.fetch_add(1, Ordering::Relaxed),
                    ));
                    buf.store(Arc::new(next));

                    next_threshold = next_threshold.saturating_add(reads_per_write);
                }

                done.store(true, Ordering::Release);
            });
        }

        for _ in 0..readers {
            let buf = buf.clone();
            let start_barrier = start_barrier.clone();
            let started = started.clone();
            let done = done.clone();
            let read_ops = read_ops.clone();

            s.spawn(move || {
                start_barrier.wait();
                while !started.load(Ordering::Acquire) {
                    std::hint::spin_loop();
                }

                while !done.load(Ordering::Acquire) {
                    let snap = buf.load_full();
                    consume_all(&snap);
                    read_ops.fetch_add(1, Ordering::Release);
                }
            });
        }
    });
}

fn run_arc_swap_vec_latency(readers: usize, ratio: usize, target_len: usize) -> LatencyResult {
    let buf: Arc<ArcSwap<Vec<Entry>>> = Arc::new(ArcSwap::from_pointee(Vec::new()));

    let start_barrier = Arc::new(std::sync::Barrier::new(readers + 1));
    let started = Arc::new(AtomicBool::new(false));
    let done = Arc::new(AtomicBool::new(false));
    let read_ops = Arc::new(AtomicUsize::new(0));
    let writer_seed = Arc::new(AtomicU32::new(0));

    std::thread::scope(|s| {
        let writer_handle = {
            let buf = buf.clone();
            let start_barrier = start_barrier.clone();
            let started = started.clone();
            let done = done.clone();
            let read_ops = read_ops.clone();
            let writer_seed = writer_seed.clone();

            s.spawn(move || {
                let mut publish_ns: Vec<u64> = Vec::with_capacity(target_len.max(1));
                start_barrier.wait();

                let t0 = Instant::now();
                let mut next = (*buf.load_full()).clone();
                next.push(Entry::from_seed(
                    writer_seed.fetch_add(1, Ordering::Relaxed),
                ));
                buf.store(Arc::new(next));
                publish_ns.push(t0.elapsed().as_nanos() as u64);
                started.store(true, Ordering::Release);

                let reads_per_write = ratio.saturating_mul(readers).max(1);
                let mut next_threshold = reads_per_write;

                loop {
                    let len = buf.load_full().len();
                    if len >= target_len {
                        break;
                    }

                    while read_ops.load(Ordering::Acquire) < next_threshold {
                        std::hint::spin_loop();
                    }

                    let t0 = Instant::now();
                    let mut next = (*buf.load_full()).clone();
                    next.push(Entry::from_seed(
                        writer_seed.fetch_add(1, Ordering::Relaxed),
                    ));
                    buf.store(Arc::new(next));
                    publish_ns.push(t0.elapsed().as_nanos() as u64);

                    next_threshold = next_threshold.saturating_add(reads_per_write);
                }

                done.store(true, Ordering::Release);
                publish_ns
            })
        };

        let mut reader_handles = Vec::with_capacity(readers);
        for _ in 0..readers {
            let buf = buf.clone();
            let start_barrier = start_barrier.clone();
            let started = started.clone();
            let done = done.clone();
            let read_ops = read_ops.clone();

            reader_handles.push(s.spawn(move || {
                let mut reader_ns: Vec<u64> = Vec::new();
                start_barrier.wait();
                while !started.load(Ordering::Acquire) {
                    std::hint::spin_loop();
                }

                while !done.load(Ordering::Acquire) {
                    let t0 = Instant::now();
                    let snap = buf.load_full();
                    consume_all(&snap);
                    reader_ns.push(t0.elapsed().as_nanos() as u64);
                    read_ops.fetch_add(1, Ordering::Release);
                }

                reader_ns
            }));
        }

        let publish_ns = writer_handle.join().unwrap();
        let mut reader_ns = Vec::new();
        for h in reader_handles {
            reader_ns.extend(h.join().unwrap());
        }

        LatencyResult {
            publish_ns,
            reader_ns,
        }
    })
}

fn run_parking_mutex_vec(readers: usize, ratio: usize, target_len: usize) {
    let buf: Arc<ParkingMutex<Vec<Entry>>> = Arc::new(ParkingMutex::new(Vec::new()));

    let start_barrier = Arc::new(std::sync::Barrier::new(readers + 1));
    let started = Arc::new(AtomicBool::new(false));
    let done = Arc::new(AtomicBool::new(false));
    let read_ops = Arc::new(AtomicUsize::new(0));
    let writer_seed = Arc::new(AtomicU32::new(0));

    std::thread::scope(|s| {
        {
            let buf = buf.clone();
            let start_barrier = start_barrier.clone();
            let started = started.clone();
            let done = done.clone();
            let read_ops = read_ops.clone();
            let writer_seed = writer_seed.clone();

            s.spawn(move || {
                start_barrier.wait();

                {
                    let mut g = buf.lock();
                    g.push(Entry::from_seed(
                        writer_seed.fetch_add(1, Ordering::Relaxed),
                    ));
                }
                started.store(true, Ordering::Release);

                let reads_per_write = ratio.saturating_mul(readers).max(1);
                let mut next_threshold = reads_per_write;

                loop {
                    let len = { buf.lock().len() };
                    if len >= target_len {
                        break;
                    }

                    while read_ops.load(Ordering::Acquire) < next_threshold {
                        std::hint::spin_loop();
                    }

                    {
                        let mut g = buf.lock();
                        g.push(Entry::from_seed(
                            writer_seed.fetch_add(1, Ordering::Relaxed),
                        ));
                    }
                    next_threshold = next_threshold.saturating_add(reads_per_write);
                }

                done.store(true, Ordering::Release);
            });
        }

        for _ in 0..readers {
            let buf = buf.clone();
            let start_barrier = start_barrier.clone();
            let started = started.clone();
            let done = done.clone();
            let read_ops = read_ops.clone();

            s.spawn(move || {
                start_barrier.wait();
                while !started.load(Ordering::Acquire) {
                    std::hint::spin_loop();
                }

                while !done.load(Ordering::Acquire) {
                    let g = buf.lock();
                    consume_all(&g);
                    drop(g);
                    read_ops.fetch_add(1, Ordering::Release);
                }
            });
        }
    });
}

fn run_parking_mutex_vec_latency(readers: usize, ratio: usize, target_len: usize) -> LatencyResult {
    let buf: Arc<ParkingMutex<Vec<Entry>>> = Arc::new(ParkingMutex::new(Vec::new()));

    let start_barrier = Arc::new(std::sync::Barrier::new(readers + 1));
    let started = Arc::new(AtomicBool::new(false));
    let done = Arc::new(AtomicBool::new(false));
    let read_ops = Arc::new(AtomicUsize::new(0));
    let writer_seed = Arc::new(AtomicU32::new(0));

    std::thread::scope(|s| {
        let writer_handle = {
            let buf = buf.clone();
            let start_barrier = start_barrier.clone();
            let started = started.clone();
            let done = done.clone();
            let read_ops = read_ops.clone();
            let writer_seed = writer_seed.clone();

            s.spawn(move || {
                let mut publish_ns: Vec<u64> = Vec::with_capacity(target_len.max(1));
                start_barrier.wait();

                let t0 = Instant::now();
                {
                    let mut g = buf.lock();
                    g.push(Entry::from_seed(
                        writer_seed.fetch_add(1, Ordering::Relaxed),
                    ));
                }
                publish_ns.push(t0.elapsed().as_nanos() as u64);
                started.store(true, Ordering::Release);

                let reads_per_write = ratio.saturating_mul(readers).max(1);
                let mut next_threshold = reads_per_write;

                loop {
                    if buf.lock().len() >= target_len {
                        break;
                    }

                    while read_ops.load(Ordering::Acquire) < next_threshold {
                        std::hint::spin_loop();
                    }

                    let t0 = Instant::now();
                    {
                        let mut g = buf.lock();
                        g.push(Entry::from_seed(
                            writer_seed.fetch_add(1, Ordering::Relaxed),
                        ));
                    }
                    publish_ns.push(t0.elapsed().as_nanos() as u64);

                    next_threshold = next_threshold.saturating_add(reads_per_write);
                }

                done.store(true, Ordering::Release);
                publish_ns
            })
        };

        let mut reader_handles = Vec::with_capacity(readers);
        for _ in 0..readers {
            let buf = buf.clone();
            let start_barrier = start_barrier.clone();
            let started = started.clone();
            let done = done.clone();
            let read_ops = read_ops.clone();

            reader_handles.push(s.spawn(move || {
                let mut reader_ns: Vec<u64> = Vec::new();
                start_barrier.wait();
                while !started.load(Ordering::Acquire) {
                    std::hint::spin_loop();
                }

                while !done.load(Ordering::Acquire) {
                    let t0 = Instant::now();
                    let g = buf.lock();
                    consume_all(&g);
                    drop(g);
                    reader_ns.push(t0.elapsed().as_nanos() as u64);
                    read_ops.fetch_add(1, Ordering::Release);
                }

                reader_ns
            }));
        }

        let publish_ns = writer_handle.join().unwrap();
        let mut reader_ns = Vec::new();
        for h in reader_handles {
            reader_ns.extend(h.join().unwrap());
        }

        LatencyResult {
            publish_ns,
            reader_ns,
        }
    })
}

fn run_parking_rwlock_vec(readers: usize, ratio: usize, target_len: usize) {
    let buf: Arc<ParkingRwLock<Vec<Entry>>> = Arc::new(ParkingRwLock::new(Vec::new()));

    let start_barrier = Arc::new(std::sync::Barrier::new(readers + 1));
    let started = Arc::new(AtomicBool::new(false));
    let done = Arc::new(AtomicBool::new(false));
    let read_ops = Arc::new(AtomicUsize::new(0));
    let writer_seed = Arc::new(AtomicU32::new(0));

    std::thread::scope(|s| {
        {
            let buf = buf.clone();
            let start_barrier = start_barrier.clone();
            let started = started.clone();
            let done = done.clone();
            let read_ops = read_ops.clone();
            let writer_seed = writer_seed.clone();

            s.spawn(move || {
                start_barrier.wait();

                {
                    let mut g = buf.write();
                    g.push(Entry::from_seed(
                        writer_seed.fetch_add(1, Ordering::Relaxed),
                    ));
                }
                started.store(true, Ordering::Release);

                let reads_per_write = ratio.saturating_mul(readers).max(1);
                let mut next_threshold = reads_per_write;

                loop {
                    let len = { buf.read().len() };
                    if len >= target_len {
                        break;
                    }

                    while read_ops.load(Ordering::Acquire) < next_threshold {
                        std::hint::spin_loop();
                    }

                    {
                        let mut g = buf.write();
                        g.push(Entry::from_seed(
                            writer_seed.fetch_add(1, Ordering::Relaxed),
                        ));
                    }
                    next_threshold = next_threshold.saturating_add(reads_per_write);
                }

                done.store(true, Ordering::Release);
            });
        }

        for _ in 0..readers {
            let buf = buf.clone();
            let start_barrier = start_barrier.clone();
            let started = started.clone();
            let done = done.clone();
            let read_ops = read_ops.clone();

            s.spawn(move || {
                start_barrier.wait();
                while !started.load(Ordering::Acquire) {
                    std::hint::spin_loop();
                }

                while !done.load(Ordering::Acquire) {
                    let g = buf.read();
                    consume_all(&g);
                    drop(g);
                    read_ops.fetch_add(1, Ordering::Release);
                }
            });
        }
    });
}

fn run_parking_rwlock_vec_latency(
    readers: usize,
    ratio: usize,
    target_len: usize,
) -> LatencyResult {
    let buf: Arc<ParkingRwLock<Vec<Entry>>> = Arc::new(ParkingRwLock::new(Vec::new()));

    let start_barrier = Arc::new(std::sync::Barrier::new(readers + 1));
    let started = Arc::new(AtomicBool::new(false));
    let done = Arc::new(AtomicBool::new(false));
    let read_ops = Arc::new(AtomicUsize::new(0));
    let writer_seed = Arc::new(AtomicU32::new(0));

    std::thread::scope(|s| {
        let writer_handle = {
            let buf = buf.clone();
            let start_barrier = start_barrier.clone();
            let started = started.clone();
            let done = done.clone();
            let read_ops = read_ops.clone();
            let writer_seed = writer_seed.clone();

            s.spawn(move || {
                let mut publish_ns: Vec<u64> = Vec::with_capacity(target_len.max(1));
                start_barrier.wait();

                let t0 = Instant::now();
                {
                    let mut g = buf.write();
                    g.push(Entry::from_seed(
                        writer_seed.fetch_add(1, Ordering::Relaxed),
                    ));
                }
                publish_ns.push(t0.elapsed().as_nanos() as u64);
                started.store(true, Ordering::Release);

                let reads_per_write = ratio.saturating_mul(readers).max(1);
                let mut next_threshold = reads_per_write;

                loop {
                    if buf.read().len() >= target_len {
                        break;
                    }

                    while read_ops.load(Ordering::Acquire) < next_threshold {
                        std::hint::spin_loop();
                    }

                    let t0 = Instant::now();
                    {
                        let mut g = buf.write();
                        g.push(Entry::from_seed(
                            writer_seed.fetch_add(1, Ordering::Relaxed),
                        ));
                    }
                    publish_ns.push(t0.elapsed().as_nanos() as u64);

                    next_threshold = next_threshold.saturating_add(reads_per_write);
                }

                done.store(true, Ordering::Release);
                publish_ns
            })
        };

        let mut reader_handles = Vec::with_capacity(readers);
        for _ in 0..readers {
            let buf = buf.clone();
            let start_barrier = start_barrier.clone();
            let started = started.clone();
            let done = done.clone();
            let read_ops = read_ops.clone();

            reader_handles.push(s.spawn(move || {
                let mut reader_ns: Vec<u64> = Vec::new();
                start_barrier.wait();
                while !started.load(Ordering::Acquire) {
                    std::hint::spin_loop();
                }

                while !done.load(Ordering::Acquire) {
                    let t0 = Instant::now();
                    let g = buf.read();
                    consume_all(&g);
                    drop(g);
                    reader_ns.push(t0.elapsed().as_nanos() as u64);
                    read_ops.fetch_add(1, Ordering::Release);
                }

                reader_ns
            }));
        }

        let publish_ns = writer_handle.join().unwrap();
        let mut reader_ns = Vec::new();
        for h in reader_handles {
            reader_ns.extend(h.join().unwrap());
        }

        LatencyResult {
            publish_ns,
            reader_ns,
        }
    })
}

fn run_std_rwlock_vec(readers: usize, ratio: usize, target_len: usize) {
    let buf: Arc<std::sync::RwLock<Vec<Entry>>> = Arc::new(std::sync::RwLock::new(Vec::new()));

    let start_barrier = Arc::new(std::sync::Barrier::new(readers + 1));
    let started = Arc::new(AtomicBool::new(false));
    let done = Arc::new(AtomicBool::new(false));
    let read_ops = Arc::new(AtomicUsize::new(0));
    let writer_seed = Arc::new(AtomicU32::new(0));

    std::thread::scope(|s| {
        {
            let buf = buf.clone();
            let start_barrier = start_barrier.clone();
            let started = started.clone();
            let done = done.clone();
            let read_ops = read_ops.clone();
            let writer_seed = writer_seed.clone();

            s.spawn(move || {
                start_barrier.wait();

                {
                    let mut g = buf.write().unwrap();
                    g.push(Entry::from_seed(
                        writer_seed.fetch_add(1, Ordering::Relaxed),
                    ));
                }
                started.store(true, Ordering::Release);

                let reads_per_write = ratio.saturating_mul(readers).max(1);
                let mut next_threshold = reads_per_write;

                loop {
                    let len = { buf.read().unwrap().len() };
                    if len >= target_len {
                        break;
                    }

                    while read_ops.load(Ordering::Acquire) < next_threshold {
                        std::hint::spin_loop();
                    }

                    {
                        let mut g = buf.write().unwrap();
                        g.push(Entry::from_seed(
                            writer_seed.fetch_add(1, Ordering::Relaxed),
                        ));
                    }
                    next_threshold = next_threshold.saturating_add(reads_per_write);
                }

                done.store(true, Ordering::Release);
            });
        }

        for _ in 0..readers {
            let buf = buf.clone();
            let start_barrier = start_barrier.clone();
            let started = started.clone();
            let done = done.clone();
            let read_ops = read_ops.clone();

            s.spawn(move || {
                start_barrier.wait();
                while !started.load(Ordering::Acquire) {
                    std::hint::spin_loop();
                }

                while !done.load(Ordering::Acquire) {
                    let g = buf.read().unwrap();
                    consume_all(&g);
                    drop(g);
                    read_ops.fetch_add(1, Ordering::Release);
                }
            });
        }
    });
}

fn run_std_rwlock_vec_latency(readers: usize, ratio: usize, target_len: usize) -> LatencyResult {
    let buf: Arc<std::sync::RwLock<Vec<Entry>>> = Arc::new(std::sync::RwLock::new(Vec::new()));

    let start_barrier = Arc::new(std::sync::Barrier::new(readers + 1));
    let started = Arc::new(AtomicBool::new(false));
    let done = Arc::new(AtomicBool::new(false));
    let read_ops = Arc::new(AtomicUsize::new(0));
    let writer_seed = Arc::new(AtomicU32::new(0));

    std::thread::scope(|s| {
        let writer_handle = {
            let buf = buf.clone();
            let start_barrier = start_barrier.clone();
            let started = started.clone();
            let done = done.clone();
            let read_ops = read_ops.clone();
            let writer_seed = writer_seed.clone();

            s.spawn(move || {
                let mut publish_ns: Vec<u64> = Vec::with_capacity(target_len.max(1));
                start_barrier.wait();

                let t0 = Instant::now();
                {
                    let mut g = buf.write().unwrap();
                    g.push(Entry::from_seed(
                        writer_seed.fetch_add(1, Ordering::Relaxed),
                    ));
                }
                publish_ns.push(t0.elapsed().as_nanos() as u64);
                started.store(true, Ordering::Release);

                let reads_per_write = ratio.saturating_mul(readers).max(1);
                let mut next_threshold = reads_per_write;

                loop {
                    if buf.read().unwrap().len() >= target_len {
                        break;
                    }

                    while read_ops.load(Ordering::Acquire) < next_threshold {
                        std::hint::spin_loop();
                    }

                    let t0 = Instant::now();
                    {
                        let mut g = buf.write().unwrap();
                        g.push(Entry::from_seed(
                            writer_seed.fetch_add(1, Ordering::Relaxed),
                        ));
                    }
                    publish_ns.push(t0.elapsed().as_nanos() as u64);

                    next_threshold = next_threshold.saturating_add(reads_per_write);
                }

                done.store(true, Ordering::Release);
                publish_ns
            })
        };

        let mut reader_handles = Vec::with_capacity(readers);
        for _ in 0..readers {
            let buf = buf.clone();
            let start_barrier = start_barrier.clone();
            let started = started.clone();
            let done = done.clone();
            let read_ops = read_ops.clone();

            reader_handles.push(s.spawn(move || {
                let mut reader_ns: Vec<u64> = Vec::new();
                start_barrier.wait();
                while !started.load(Ordering::Acquire) {
                    std::hint::spin_loop();
                }

                while !done.load(Ordering::Acquire) {
                    let t0 = Instant::now();
                    let g = buf.read().unwrap();
                    consume_all(&g);
                    drop(g);
                    reader_ns.push(t0.elapsed().as_nanos() as u64);
                    read_ops.fetch_add(1, Ordering::Release);
                }

                reader_ns
            }));
        }

        let publish_ns = writer_handle.join().unwrap();
        let mut reader_ns = Vec::new();
        for h in reader_handles {
            reader_ns.extend(h.join().unwrap());
        }

        LatencyResult {
            publish_ns,
            reader_ns,
        }
    })
}
