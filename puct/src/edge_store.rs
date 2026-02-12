use std::cmp::Ordering;

use append_only_vec::AppendOnlyVec;
use model::ActionWithPolicy;

use super::{Comparator, InPlaceMaxHeap, PUCTEdge};

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
        if self.heap.remaining() == 0 {
            return;
        }
        self.heap.extract_next_to_end();
        self.edges.push(PUCTEdge::new());
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
