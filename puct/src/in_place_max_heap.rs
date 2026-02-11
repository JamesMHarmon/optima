use std::cmp::Ordering;

pub(crate) trait Comparator<T> {
    fn cmp(&self, a: &T, b: &T) -> Ordering;
}

#[derive(Clone, Copy, Default)]
pub(crate) struct OrdComparator;

impl<T: Ord> Comparator<T> for OrdComparator {
    #[inline]
    fn cmp(&self, a: &T, b: &T) -> Ordering {
        a.cmp(b)
    }
}

/// Generic in-place max-heap over a `Vec<T>`.
///
/// The heap occupies `items[0..heap_len]`. Each `extract_next_to_end()` moves the current maximum
/// to the end of the heap region, shrinks the heap boundary, and restores the heap property.
pub(crate) struct InPlaceMaxHeap<T, C = OrdComparator> {
    heap_len: usize,
    items: Box<[T]>,
    cmp: C,
}

impl<T: Ord> InPlaceMaxHeap<T, OrdComparator> {
    pub(crate) fn new(items: Box<[T]>) -> Self {
        Self::with_comparator(items, OrdComparator)
    }
}

impl<T, C: Comparator<T>> InPlaceMaxHeap<T, C> {
    pub(crate) fn with_comparator(items: Box<[T]>, cmp: C) -> Self {
        let heap_len = items.len();
        Self {
            heap_len,
            items,
            cmp,
        }
    }

    pub(crate) fn remaining(&self) -> usize {
        self.heap_len
    }

    pub(crate) fn total_len(&self) -> usize {
        self.items.len()
    }

    pub(crate) fn items(&self) -> &[T] {
        &self.items
    }

    /// Heapify `items[0..heap_len]` into a max-heap.
    pub(crate) fn heapify_max(&mut self) {
        let heap_len = self.heap_len;
        if heap_len <= 1 {
            return;
        }

        let mut i = heap_len / 2;
        while i > 0 {
            i -= 1;
            self.sift_down_max(i, heap_len);
        }
    }

    /// Extract the current maximum to the end of the heap region.
    ///
    /// After `k` extractions, the last `k` elements of the original array are the `k` best items
    /// in descending order.
    pub(crate) fn extract_next_to_end(&mut self) {
        if self.heap_len == 0 {
            return;
        }

        let last = self.heap_len - 1;
        self.items.swap(0, last);
        self.heap_len -= 1;
        if self.heap_len > 0 {
            let heap_len = self.heap_len;
            self.sift_down_max(0, heap_len);
        }
    }

    fn sift_down_max(&mut self, mut root: usize, heap_len: usize) {
        loop {
            let left = root * 2 + 1;
            if left >= heap_len {
                return;
            }

            let right = left + 1;
            let mut best = left;

            if right < heap_len {
                if self.cmp.cmp(&self.items[right], &self.items[left]) == Ordering::Greater {
                    best = right;
                }
            }

            if self.cmp.cmp(&self.items[best], &self.items[root]) != Ordering::Greater {
                return;
            }

            self.items.swap(root, best);
            root = best;
        }
    }
}
