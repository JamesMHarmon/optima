# rcu-append-buffer

A single-writer, multi-reader, append-only buffer with RCU-style snapshots.

## Properties

- Readers take no locks: `snapshot()` is an atomic load + length capture.
- Writer appends in-place until capacity, then grows by copy-on-grow.
- Snapshots are contiguous slices (`&[T]`) suitable for fast scans.
- Snapshot views are immutable and stable for the lifetime of the snapshot.
- Debug builds validate ΓÇ£single writer threadΓÇ¥ (best-effort).

## Guarantees (assuming exactly one writer)

- Snapshots are frozen at creation (length captured).
- Published elements are never mutated; growth clones into a new buffer.
- Old buffers remain valid until the last snapshot referencing them is dropped.
- No dynamic dispatch.
- Fast path (`push` without growth) does not clone.

## Constraints

- Exactly one writer thread. Violating this is undefined behavior.
- `T: Clone + Send + Sync` (growth clones existing elements).

## Example

```rust
use collections::RcuAppendBuffer;

let buffer = RcuAppendBuffer::new();

buffer.push(1);
buffer.push(2);

let snap = buffer.snapshot();
assert_eq!(snap.as_slice(), &[1, 2]);
```

## Validation

From the workspace root:

- `cargo test -p collections`
- Optional (nightly): `cargo +nightly miri test -p collections`
- Optional (nightly + supported toolchain): `RUSTFLAGS="-Z sanitizer=thread" cargo +nightly test -p collections`
- `cargo test -p collections --release`

## Intended Use

High-read, low-write scenarios where:

- Reads dominate.
- Snapshot consistency matters.
- Append-only semantics are sufficient.

## Performance Notes

Criterion benchmarks live in `collections/benches/rcu_append_buffer.rs` and model a PUCT-like workload (many full scans between appends).

- `RcuAppendBuffer`: very fast reads (mean ~0.09┬╡s per full scan) with low publish cost (mean ~0.34┬╡s).
- `AppendOnlyVec`: fastest publish/append (mean ~0.16┬╡s) with similar read cost at this length (mean ~0.11┬╡s).
- `parking_lot::RwLock<Vec<_>>`: competitive in this low-reader case (publish mean ~0.25┬╡s, read mean ~0.11┬╡s), but readers take a lock and can block the writer.
- `ArcSwap<Vec<_>>`: much higher publish cost (mean ~2.36┬╡s) and large tail latency spikes (p99 ~51┬╡s), due to copy-on-write cloning.

Reproduce (from workspace root): `OPTIMA_LATENCY=1 cargo bench -p collections --bench rcu_append_buffer`
