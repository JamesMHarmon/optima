# rcu-append-buffer

A single-writer, multi-reader, append-only buffer with RCU-style snapshots.

## Properties

- Readers take no locks: `snapshot()` is an atomic load + length capture.
- Writer appends in-place until capacity, then grows by copy-on-grow.
- Snapshots are contiguous slices (`&[T]`) suitable for fast scans.
- Snapshot views are immutable and stable for the lifetime of the snapshot.
- Debug builds validate “single writer thread” (best-effort).

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
use rcu_append_buffer::RcuAppendBuffer;

let buffer = RcuAppendBuffer::new();

buffer.push(1);
buffer.push(2);

let snap = buffer.snapshot();
assert_eq!(snap.as_slice(), &[1, 2]);
```

## Validation

From the workspace root:

- `cargo test -p rcu-append-buffer`
- Optional (nightly): `cargo +nightly miri test -p rcu-append-buffer`
- Optional (nightly + supported toolchain): `RUSTFLAGS="-Z sanitizer=thread" cargo +nightly test -p rcu-append-buffer`
- `cargo test -p rcu-append-buffer --release`

## Intended Use

High-read, low-write scenarios where:

- Reads dominate.
- Snapshot consistency matters.
- Append-only semantics are sufficient.