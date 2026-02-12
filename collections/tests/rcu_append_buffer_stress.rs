use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

use collections::RcuAppendBuffer;

#[test]
fn stress_push_and_snapshot() {
    use std::thread;

    const TOTAL: usize = 2000;

    let buffer = Arc::new(RcuAppendBuffer::new());

    let constructed = Arc::new(AtomicUsize::new(0));
    let dropped = Arc::new(AtomicUsize::new(0));

    let writer_buffer = buffer.clone();
    let writer_constructed = constructed.clone();
    let writer_dropped = dropped.clone();

    let writer = thread::spawn(move || {
        for i in 0..TOTAL {
            writer_buffer.push(DropCounter::new(
                i,
                writer_constructed.clone(),
                writer_dropped.clone(),
            ));
            if i % 7 == 0 {
                thread::yield_now();
            }
        }
    });

    let reader_buffer = buffer.clone();

    let reader = thread::spawn(move || {
        let mut last_len = 0;
        while last_len < TOTAL {
            let snap = reader_buffer.snapshot();
            let slice = snap.as_slice();

            assert!(slice.len() >= last_len);
            last_len = slice.len();

            for (i, item) in slice.iter().enumerate() {
                assert_eq!(item.value, i);
            }

            thread::yield_now();
        }
    });

    writer.join().unwrap();
    reader.join().unwrap();

    drop(buffer);

    let constructed_count = constructed.load(Ordering::Relaxed);
    let dropped_count = dropped.load(Ordering::Relaxed);

    assert_eq!(
        constructed_count, dropped_count,
        "Every constructed instance must be dropped exactly once"
    );
}

#[test]
fn stress_snapshot_cloning() {
    const TOTAL: usize = 512;

    let buffer = Arc::new(RcuAppendBuffer::new());

    for i in 0..TOTAL {
        buffer.push(i);
    }

    let mut handles = vec![];

    for _ in 0..8 {
        let b = buffer.clone();
        handles.push(thread::spawn(move || {
            for _ in 0..10_000 {
                let snap = b.snapshot();
                let slice = snap.as_slice();
                for (i, val) in slice.iter().enumerate() {
                    assert_eq!(*val, i);
                }
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }
}

#[test]
fn growth_tiers_test() {
    const TOTAL: usize = 2000;

    let buffer = RcuAppendBuffer::new();

    for i in 0..TOTAL {
        buffer.push(i);
    }

    let snap = buffer.snapshot();
    let slice = snap.as_slice();

    assert_eq!(slice.len(), TOTAL);

    for (i, v) in slice.iter().enumerate() {
        assert_eq!(*v, i);
    }
}

#[test]
fn old_snapshot_retention() {
    let buffer = Arc::new(RcuAppendBuffer::new());

    for i in 0..100 {
        buffer.push(i);
    }

    let old_snapshot = buffer.snapshot();

    for i in 100..500 {
        buffer.push(i);
    }

    let new_snapshot = buffer.snapshot();

    assert_eq!(old_snapshot.as_slice().len(), 100);
    assert_eq!(new_snapshot.as_slice().len(), 500);

    for (i, v) in old_snapshot.as_slice().iter().enumerate() {
        assert_eq!(*v, i);
    }

    for (i, v) in new_snapshot.as_slice().iter().enumerate() {
        assert_eq!(*v, i);
    }
}

#[test]
fn snapshot_is_frozen_during_resize() {
    let buffer = RcuAppendBuffer::new();

    for i in 0..100 {
        buffer.push(i);
    }

    let snap = buffer.snapshot();

    for i in 100..500 {
        buffer.push(i);
    }

    let slice = snap.as_slice();

    assert_eq!(slice.len(), 100);

    for (i, v) in slice.iter().enumerate() {
        assert_eq!(*v, i);
    }
}

#[test]
fn concurrent_resize_stress() {
    use std::sync::Arc;
    use std::thread;

    let buffer = Arc::new(RcuAppendBuffer::new());

    let writer = {
        let b = buffer.clone();
        thread::spawn(move || {
            for i in 0..2000 {
                b.push(i);
            }
        })
    };

    let mut readers = vec![];

    for _ in 0..8 {
        let b = buffer.clone();
        readers.push(thread::spawn(move || {
            for _ in 0..10_000 {
                let snap = b.snapshot();
                let slice = snap.as_slice();
                for (i, v) in slice.iter().enumerate() {
                    assert_eq!(*v, i);
                }
            }
        }));
    }

    writer.join().unwrap();
    for r in readers {
        r.join().unwrap();
    }
}

#[test]
fn old_buffers_drop_after_last_snapshot() {
    let buffer = RcuAppendBuffer::new();

    for i in 0..2000 {
        buffer.push(i);
    }

    let snap = buffer.snapshot();

    drop(buffer);

    assert_eq!(snap.as_slice().len(), 2000);
}

#[test]
fn no_mutation_of_published_prefix() {
    let buffer = RcuAppendBuffer::new();

    for i in 0..64 {
        buffer.push(i);
    }

    let snap = buffer.snapshot();
    let before = snap.as_slice().to_vec();

    for i in 64..128 {
        buffer.push(i);
    }

    let after = snap.as_slice().to_vec();

    assert_eq!(before, after);
}

struct DropCounter {
    value: usize,
    constructed: Arc<AtomicUsize>,
    dropped: Arc<AtomicUsize>,
}

impl DropCounter {
    fn new(value: usize, constructed: Arc<AtomicUsize>, dropped: Arc<AtomicUsize>) -> Self {
        constructed.fetch_add(1, Ordering::Relaxed);
        Self {
            value,
            constructed,
            dropped,
        }
    }
}

impl Drop for DropCounter {
    fn drop(&mut self) {
        self.dropped.fetch_add(1, Ordering::Relaxed);
    }
}

impl Clone for DropCounter {
    fn clone(&self) -> Self {
        self.constructed.fetch_add(1, Ordering::Relaxed);
        Self {
            value: self.value,
            constructed: self.constructed.clone(),
            dropped: self.dropped.clone(),
        }
    }
}
