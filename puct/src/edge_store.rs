use std::cmp::Ordering;

use append_only_vec::AppendOnlyVec;
use model::ActionWithPolicy;
use parking_lot::{RwLock, RwLockReadGuard};

use super::{Comparator, InPlaceMaxHeap, PUCTEdge};

type PolicyPriorsHeap<A> = InPlaceMaxHeap<ActionWithPolicy<A>, PolicyScoreCmp>;

/// In-place heapified edge store (heap over `policy_priors` themselves), guarded by an `RwLock`.
///
/// This is intended for benchmarks: it avoids allocating a separate heap of indices.
///
/// Notes:
/// - Reorders `policy_priors` in-place under an `RwLock` write lock.
/// - Heapifies priors once, then extracts the next-best prior by moving it to the end.
/// - `action_idx` is implicit and maps `edge_index i` to `policy_priors[len - 1 - i]`.
pub(crate) struct EdgeStore<A> {
    edges: AppendOnlyVec<PUCTEdge>,
    heap: RwLock<PolicyPriorsHeap<A>>,
}

type ActionWithPolicyGuard<'a, A> = parking_lot::MappedRwLockReadGuard<'a, ActionWithPolicy<A>>;

impl<A> EdgeStore<A> {
    pub(crate) fn new(policy_priors: Box<[ActionWithPolicy<A>]>) -> Self {
        Self {
            edges: AppendOnlyVec::new(),
            heap: RwLock::new(InPlaceMaxHeap::with_comparator(
                policy_priors,
                PolicyScoreCmp,
            )),
        }
    }

    #[inline]
    fn heap_initialized(&self) -> bool {
        self.edges.len() != 0
    }

    fn action_with_policy(&self, edge_index: usize) -> ActionWithPolicyGuard<'_, A> {
        let heap = self.heap.read();

        RwLockReadGuard::map(heap, |v| {
            let total_len = v.total_len();
            debug_assert!(edge_index < total_len);
            let idx = Self::phys(total_len, edge_index);
            &v.items()[idx]
        })
    }

    pub(crate) fn edges_iter(
        &self,
    ) -> impl DoubleEndedIterator<Item = (&PUCTEdge, ActionWithPolicyGuard<'_, A>)> + ExactSizeIterator
    {
        self.edges.iter().enumerate().map(move |(i, edge)| {
            let action_with_policy = self.action_with_policy(i);
            (edge, action_with_policy)
        })
    }

    pub(crate) fn edge(&self, index: usize) -> (&PUCTEdge, ActionWithPolicyGuard<'_, A>) {
        let edge = &self.edges[index];
        let action_with_policy = self.action_with_policy(index);
        (edge, action_with_policy)
    }

    pub(crate) fn ensure_frontier_edge(&self) {
        if self.has_unvisited_frontier_edge() {
            return;
        }

        let mut heap = self.heap.write();

        self.materialize_next_edge(&mut heap);
    }

    #[inline]
    fn has_unvisited_frontier_edge(&self) -> bool {
        let edge_count = self.edges.len();
        if edge_count == 0 {
            return false;
        }
        self.edges[edge_count - 1].visits() == 0
    }

    #[inline]
    fn initialize_heap_if_first_edge(&self, heap: &mut PolicyPriorsHeap<A>) {
        if self.heap_initialized() {
            return;
        }

        heap.heapify_max();
    }

    #[inline]
    fn materialize_next_edge(&self, heap: &mut PolicyPriorsHeap<A>) {
        if heap.remaining() == 0 {
            return;
        }

        self.initialize_heap_if_first_edge(heap);
        heap.extract_next_to_end();
        self.edges.push(PUCTEdge::new());
    }

    #[inline]
    fn phys(total_len: usize, edge_index: usize) -> usize {
        total_len - 1 - edge_index
    }
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
