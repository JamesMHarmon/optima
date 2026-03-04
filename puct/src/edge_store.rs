use std::cmp::Ordering;

use append_only_vec::AppendOnlyVec;
use model::ActionWithPolicy;

use collections::{Comparator, InPlaceMaxHeap};

use crate::edge::PUCTEdge;

type PolicyPriorsHeap<A> = InPlaceMaxHeap<ActionWithPolicy<A>, PolicyScoreCmp>;

pub(crate) struct EdgeStore<A> {
    edges: AppendOnlyVec<PUCTEdge>,
    heap: PolicyPriorsHeap<A>,
}

pub(crate) type ActionWithPolicyGuard<'a, A> = &'a ActionWithPolicy<A>;

impl<A> EdgeStore<A> {
    pub(crate) fn new(policy_priors: Box<[ActionWithPolicy<A>]>) -> Self {
        Self {
            edges: AppendOnlyVec::new(),
            heap: InPlaceMaxHeap::with_comparator(policy_priors, PolicyScoreCmp),
        }
    }

    #[inline]
    fn action_with_policy(&self, edge_index: usize) -> ActionWithPolicyGuard<'_, A> {
        self.heap.extracted(edge_index)
    }

    pub(crate) fn iter_edges(&self) -> impl Iterator<Item = &PUCTEdge> {
        self.edges.iter()
    }

    pub(crate) fn iter_edges_with_policy(
        &self,
    ) -> impl Iterator<Item = (&PUCTEdge, ActionWithPolicyGuard<'_, A>)> {
        self.iter_edges().zip(self.heap.extracted_iter())
    }

    pub(crate) fn edge(&self, index: usize) -> (&PUCTEdge, ActionWithPolicyGuard<'_, A>) {
        let edge = &self.edges[index];
        let action_with_policy = self.action_with_policy(index);
        (edge, action_with_policy)
    }

    pub(crate) fn num_actions(&self) -> usize {
        self.heap.total_len()
    }

    /// Resets all edges by passing the full set of `ActionWithPolicy` items to `f` for
    /// mutation, then rebuilding the store from scratch with the updated priors.
    /// All existing edge state (visits, children) is discarded.
    pub(crate) fn reset_edges(&mut self, f: impl FnOnce(&mut [ActionWithPolicy<A>])) {
        let temp = InPlaceMaxHeap::with_comparator(Box::new([]), PolicyScoreCmp);
        let mut items = std::mem::replace(&mut self.heap, temp).into_items();
        f(&mut items);
        *self = EdgeStore::new(items);
    }

    pub(crate) fn ensure_frontier_edge(&self) {
        if self.has_unvisited_frontier_edge() {
            return;
        }

        self.materialize_next_edge();
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
    fn materialize_next_edge(&self) {
        if self.heap.remaining() > 0
            && let Some(_) = self.heap.extract_next()
        {
            self.edges.push(PUCTEdge::new());
        }
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
