use std::cell::UnsafeCell;
use std::cmp::Ordering;
use std::ptr;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

use parking_lot::Mutex;

pub trait Comparator<T> {
    fn cmp(&self, a: &T, b: &T) -> Ordering;
}

#[derive(Clone, Copy, Default)]
pub struct OrdComparator;

impl<T: Ord> Comparator<T> for OrdComparator {
    #[inline]
    fn cmp(&self, a: &T, b: &T) -> Ordering {
        a.cmp(b)
    }
}

/// In-place max-heap with a stable tail.
///
/// Invariants:
/// - Heap (unstable, mutable):      [0 .. heap_len)
/// - Stable (immutable forever):    [heap_len .. n)
///
/// Each extraction moves the max element into index `heap_len - 1`, then decreases `heap_len`,
/// thereby *growing* the stable region by 1.
/// Stable region is in descending order from the end toward the front:
/// - extracted(0) is the first extracted (largest), at index n-1
/// - extracted(1) at index n-2, etc.
pub struct InPlaceMaxHeap<T, C = OrdComparator> {
    heap_len: AtomicUsize,
    writer: Mutex<()>,
    items: Box<[UnsafeCell<T>]>,
    cmp: C,
}

impl<T: Ord> InPlaceMaxHeap<T, OrdComparator> {
    pub fn new(items: Box<[T]>) -> Self {
        Self::with_comparator(items, OrdComparator)
    }
}

impl<T, C: Comparator<T>> InPlaceMaxHeap<T, C> {
    pub fn with_comparator(items: Box<[T]>, cmp: C) -> Self {
        let heap_len = items.len();
        let items = Self::into_unsafe_cells(items);

        Self {
            heap_len: AtomicUsize::new(heap_len),
            writer: Mutex::new(()),
            items,
            cmp,
        }
    }

    #[inline]
    pub fn remaining(&self) -> usize {
        self.heap_len.load(AtomicOrdering::Acquire)
    }

    #[inline]
    pub fn total_len(&self) -> usize {
        self.items.len()
    }

    /// Number of elements that have been extracted into the stable region.
    #[inline]
    pub fn extracted_len(&self) -> usize {
        self.total_len() - self.remaining()
    }

    /// Returns the i-th extracted (stable) item as a shared reference.
    ///
    /// `i = 0` is the first extracted (largest), `i = 1` the next, etc.
    ///
    /// Safety reasoning:
    /// - We load `heap_len` with Acquire.
    /// - We only ever return a reference to an index `>= heap_len` (stable region).
    /// - Writers only mutate indices `< heap_len` (heap region) and only decrease `heap_len`.
    /// - The writer publishes the newly-stable slot *before* decreasing `heap_len` via Release store.
    pub fn extracted(&self, i: usize) -> &T {
        let heap_len = self.heap_len.load(AtomicOrdering::Acquire);
        let total = self.items.len();
        let stable_count = total - heap_len;

        assert!(i < stable_count, "extracted index out of range");

        let idx = total - 1 - i;
        debug_assert!(idx >= heap_len);

        // SAFETY: idx is in the stable region, which is never mutated again.
        unsafe { &*self.items[idx].get() }
    }

    /// Returns a slice of all stable items.
    /// The order is from the end of the array toward the front: first extracted (largest) is last.
    pub fn stable_slice(&self) -> &[T] {
        let start = self.heap_len.load(AtomicOrdering::Acquire);
        let cells: &[UnsafeCell<T>] = &self.items[start..];
        unsafe { std::slice::from_raw_parts(cells.as_ptr() as *const T, cells.len()) }
    }

    /// A convenience iterator over extracted (stable) items, in extraction order:
    /// largest first.
    pub fn extracted_iter(&self) -> impl Iterator<Item = &T> {
        self.stable_slice().iter().rev()
    }

    /// Extract the current maximum into the stable region (to the end of the array).
    ///
    /// This grows the stable region by 1.
    ///
    /// Enforced single-writer. Safe with concurrent stable reads because:
    /// - We write the new stable slot (via swap)
    /// - Then publish the new boundary with Release store
    /// - Readers use Acquire load before reading stable slots
    pub fn extract_next_to_end(&self) {
        self.with_writer(|| {
            let _ = self.extract_next_index_writer();
        })
    }

    /// Extract next max and return a reference to the newly-stable element.
    ///
    /// This is useful for "driving" extraction without exposing unstable memory.
    pub fn extract_next(&self) -> Option<&T> {
        self.with_writer(|| {
            let idx = self.extract_next_index_writer()?;
            // SAFETY: `idx` is in the stable region, which is never mutated again.
            Some(unsafe { &*self.items[idx].get() })
        })
    }

    // ----------------- internal (writer-only) helpers -----------------

    /// Heapify the heap region `[0..heap_len)` into a max-heap.
    ///
    /// MUST be called while holding the writer guard.
    fn heapify_max_writer(&self, heap_len: usize) {
        if heap_len <= 1 {
            return;
        }

        let mut i = heap_len / 2;
        while i > 0 {
            i -= 1;
            self.sift_down_max(i, heap_len);
        }
    }

    fn sift_down_max(&self, mut root: usize, heap_len: usize) {
        loop {
            let left = root * 2 + 1;
            if left >= heap_len {
                return;
            }

            let right = left + 1;
            let mut best = left;

            if right < heap_len {
                // Writer-only: safe to take shared refs into heap region temporarily
                // because this function is only called while holding the writer guard.
                let ord = self.cmp.cmp(self.heap_ref(right), self.heap_ref(left));
                if ord == Ordering::Greater {
                    best = right;
                }
            }

            if self.cmp.cmp(self.heap_ref(best), self.heap_ref(root)) != Ordering::Greater {
                return;
            }

            self.swap_items(root, best);
            root = best;
        }
    }

    /// Performs one extraction and returns the index of the newly-stable element.
    ///
    /// MUST be called while holding the writer guard.
    #[inline]
    fn extract_next_index_writer(&self) -> Option<usize> {
        let heap_len = self.heap_len.load(AtomicOrdering::Acquire);
        if heap_len == 0 {
            return None;
        }

        // Lazy init: the heap is only guaranteed to be heapified once extraction starts.
        // We treat "no extractions yet" as `heap_len == total_len`.
        if heap_len == self.total_len() {
            self.heapify_max_writer(heap_len);
        }

        let last = heap_len - 1;

        // Write the soon-to-be-stable slot `last`.
        self.swap_items(0, last);

        // Publish the new stable boundary. Slots >= last are stable.
        self.heap_len.store(last, AtomicOrdering::Release);

        // Restore heap property inside the reduced heap region [0..last).
        if last > 0 {
            self.sift_down_max(0, last);
        }

        Some(last)
    }

    #[inline]
    fn swap_items(&self, a: usize, b: usize) {
        // Writer-only: the writer guard enforces exclusive heap mutation.
        unsafe {
            ptr::swap(self.items[a].get(), self.items[b].get());
        }
    }

    #[inline]
    fn heap_ref(&self, idx: usize) -> &T {
        // This is ONLY valid while holding the writer guard (exclusive mutation).
        // It must never escape beyond the mutation step that might move elements.
        unsafe { &*self.items[idx].get() }
    }

    #[inline]
    fn with_writer<R>(&self, f: impl FnOnce() -> R) -> R {
        let _w = self.writer.lock();
        f()
    }

    fn into_unsafe_cells(items: Box<[T]>) -> Box<[UnsafeCell<T>]> {
        let raw: *mut [T] = Box::into_raw(items);
        // SAFETY: `UnsafeCell<T>` is `#[repr(transparent)]` over `T`, so `[T]` and
        // `[UnsafeCell<T>]` have identical layout and slice metadata.
        unsafe { Box::from_raw(raw as *mut [UnsafeCell<T>]) }
    }
}

// ---- Send/Sync ----
//
// We hand out `&T` from `&self` across threads (stable region), so we require `T: Sync` for Sync.
// Mutations are internally single-writer gated; stable region is disjoint from writes.

unsafe impl<T: Send, C: Send> Send for InPlaceMaxHeap<T, C> {}
unsafe impl<T: Sync, C: Sync> Sync for InPlaceMaxHeap<T, C> {}
