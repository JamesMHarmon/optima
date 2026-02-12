use std::cmp::Ordering;
use std::sync::Arc;

use collections::{Comparator, InPlaceMaxHeap};

fn boxed<T>(v: Vec<T>) -> Box<[T]> {
    v.into_boxed_slice()
}

fn extract_all_to_end<T: Ord + Clone>(heap: &InPlaceMaxHeap<T>) -> Vec<T> {
    let mut out = Vec::with_capacity(heap.total_len());
    while heap.remaining() > 0 {
        let v = heap.extract_next().expect("remaining() > 0 implies Some");
        out.push(v.clone());
    }
    out
}

#[test]
fn empty_heap_behaves() {
    let heap: InPlaceMaxHeap<u32> = InPlaceMaxHeap::new(boxed(vec![]));

    assert_eq!(heap.total_len(), 0);
    assert_eq!(heap.remaining(), 0);
    assert_eq!(heap.extracted_len(), 0);
    assert!(heap.stable_slice().is_empty());
    assert_eq!(heap.extracted_iter().count(), 0);
    assert!(heap.extract_next().is_none());
}

#[test]
#[should_panic]
fn extracted_on_empty_panics() {
    let heap: InPlaceMaxHeap<u32> = InPlaceMaxHeap::new(boxed(vec![]));
    let _ = heap.extracted(0);
}

#[test]
fn single_element_extraction() {
    let heap = InPlaceMaxHeap::new(boxed(vec![42u32]));

    assert_eq!(heap.remaining(), 1);
    assert_eq!(heap.extracted_len(), 0);

    let v = heap.extract_next().copied();
    assert_eq!(v, Some(42));

    assert_eq!(heap.remaining(), 0);
    assert_eq!(heap.extracted_len(), 1);

    assert_eq!(heap.stable_slice(), &[42]);
    assert_eq!(heap.extracted(0), &42);
    assert!(heap.extract_next().is_none());
}

#[test]
fn extracted_and_iter_orders_match() {
    let heap = InPlaceMaxHeap::new(boxed(vec![3u32, 1, 4, 2]));

    heap.extract_next_to_end();
    heap.extract_next_to_end();

    // After two extractions, stable prefix holds the 2 largest values in extraction order.
    assert_eq!(heap.stable_slice(), &[4, 3]);

    // extracted(i) is in extraction order (largest first).
    assert_eq!(*heap.extracted(0), 4);
    assert_eq!(*heap.extracted(1), 3);

    let iter: Vec<u32> = heap.extracted_iter().copied().collect();
    assert_eq!(iter, vec![4, 3]);
}

#[test]
#[should_panic]
fn extracted_out_of_range_panics() {
    let heap = InPlaceMaxHeap::new(boxed(vec![3u32, 1, 4, 2]));
    heap.extract_next_to_end();
    heap.extract_next_to_end();
    let _ = heap.extracted(2);
}

#[test]
fn full_extraction_produces_sorted_stable_slice() {
    let heap = InPlaceMaxHeap::new(boxed(vec![9u32, 1, 8, 2, 7, 3, 6, 4, 5]));

    let extracted = extract_all_to_end(&heap);
    assert_eq!(extracted, vec![9, 8, 7, 6, 5, 4, 3, 2, 1]);

    // After full extraction, stable_slice covers the entire array.
    assert_eq!(heap.remaining(), 0);
    assert_eq!(heap.extracted_len(), heap.total_len());

    // Stable slice is the whole array, in extraction order (descending).
    assert_eq!(heap.stable_slice(), &[9, 8, 7, 6, 5, 4, 3, 2, 1]);

    // extracted_iter() matches stable_slice().
    let iter: Vec<u32> = heap.extracted_iter().copied().collect();
    assert_eq!(iter, vec![9, 8, 7, 6, 5, 4, 3, 2, 1]);
}

#[test]
fn stable_tail_after_partial_extraction_contains_top_k_sorted() {
    let heap = InPlaceMaxHeap::new(boxed(vec![10u32, 3, 7, 1, 9, 2, 8, 4, 6, 5]));

    // Extract 4 items.
    for _ in 0..4 {
        heap.extract_next_to_end();
    }

    assert_eq!(heap.remaining(), 6);
    assert_eq!(heap.extracted_len(), 4);

    // Stable prefix should be the top-4 elements in extraction order.
    assert_eq!(heap.stable_slice(), &[10, 9, 8, 7]);

    // And extracted order is descending.
    let extracted: Vec<u32> = heap.extracted_iter().copied().collect();
    assert_eq!(extracted, vec![10, 9, 8, 7]);
}

#[test]
fn extracted_reference_stays_valid_across_more_extractions() {
    let heap = InPlaceMaxHeap::new(boxed(vec![5u32, 1, 4, 2, 3]));

    let first = heap.extract_next().expect("non-empty");
    assert_eq!(*first, 5);

    // Keep a raw pointer (stable region should never move/mutate).
    let first_ptr: *const u32 = first as *const u32;

    // Extract more.
    heap.extract_next_to_end();
    heap.extract_next_to_end();

    // Reference still points at the original value.
    unsafe {
        assert_eq!(*first_ptr, 5);
    }

    // And extracted(0) is still the largest (first extracted).
    assert_eq!(*heap.extracted(0), 5);
}

#[test]
fn concurrent_extractions_complete_and_sort() {
    let heap = Arc::new(InPlaceMaxHeap::new(boxed((0u32..1000).collect())));

    // Multiple threads extract; internal Mutex serializes writes.
    std::thread::scope(|s| {
        for _ in 0..8 {
            let h = heap.clone();
            s.spawn(move || {
                // Keep extracting until empty.
                loop {
                    if h.remaining() == 0 {
                        break;
                    }
                    h.extract_next_to_end();
                }
            });
        }
    });

    assert_eq!(heap.remaining(), 0);
    assert_eq!(heap.extracted_len(), heap.total_len());

    // After full extraction, stable_slice should be sorted descending (extraction order).
    let slice = heap.stable_slice();
    assert_eq!(slice.len(), 1000);
    for w in slice.windows(2) {
        assert!(w[0] >= w[1]);
    }

    // Sanity: extracted_iter is descending.
    let first = heap.extracted_iter().next().copied();
    assert_eq!(first, Some(999));
}

#[test]
fn custom_comparator_orders_by_key() {
    #[derive(Clone, Debug, PartialEq, Eq)]
    struct Item {
        key: u32,
        payload: u32,
    }

    struct KeyCmp;

    impl Comparator<Item> for KeyCmp {
        fn cmp(&self, a: &Item, b: &Item) -> Ordering {
            a.key.cmp(&b.key)
        }
    }

    let items = boxed(vec![
        Item {
            key: 2,
            payload: 20,
        },
        Item {
            key: 1,
            payload: 10,
        },
        Item {
            key: 4,
            payload: 40,
        },
        Item {
            key: 3,
            payload: 30,
        },
    ]);

    let heap = InPlaceMaxHeap::with_comparator(items, KeyCmp);

    // Drive full extraction.
    let mut keys = Vec::new();
    while let Some(it) = heap.extract_next() {
        keys.push(it.key);
    }

    assert_eq!(keys, vec![4, 3, 2, 1]);

    // Stable slice is in extraction order (descending by key).
    let stable_keys: Vec<u32> = heap.stable_slice().iter().map(|it| it.key).collect();
    assert_eq!(stable_keys, vec![4, 3, 2, 1]);
}
