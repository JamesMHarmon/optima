use append_only_vec::AppendOnlyVec;
use model::ActionWithPolicy;

use super::PUCTEdge;

pub(crate) struct EdgeStore<A> {
    policy_priors: Box<[ActionWithPolicy<A>]>,
    edges: AppendOnlyVec<PUCTEdge>,
}

impl<A> EdgeStore<A> {
    pub(crate) fn new(policy_priors: Box<[ActionWithPolicy<A>]>) -> Self {
        Self {
            policy_priors,
            edges: AppendOnlyVec::new(),
        }
    }

    pub(crate) fn policy_priors(&self) -> &[ActionWithPolicy<A>] {
        &self.policy_priors
    }

    pub(crate) fn edges_iter(
        &self,
    ) -> impl DoubleEndedIterator<Item = &PUCTEdge> + ExactSizeIterator {
        self.edges.iter()
    }

    pub(crate) fn get_edge(&self, index: usize) -> &PUCTEdge {
        &self.edges[index]
    }

    pub(crate) fn edge_count(&self) -> usize {
        self.edges.len()
    }

    pub(crate) fn iter_edge_refs(
        &self,
    ) -> impl DoubleEndedIterator<Item = &PUCTEdge> + ExactSizeIterator {
        self.edges.iter()
    }

    pub(crate) fn get_action(&self, action_idx: u32) -> &ActionWithPolicy<A> {
        &self.policy_priors[action_idx as usize]
    }

    /// Ensures there is at most one frontier edge (defined as `visits == 0`), and that if there
    /// is no frontier edge (i.e. the last edge has `visits > 0`), a new one is materialized with
    /// the highest policy prior among not-yet-materialized actions.
    pub(crate) fn ensure_frontier_edge(&self) {
        let edge_count = self.edges.len();
        if edge_count != 0 {
            let last_edge = &self.edges[edge_count - 1];
            if last_edge.visits() == 0 {
                return;
            }
        }

        let mut best_action_idx: Option<u32> = None;
        let mut best_score = None;

        for (idx, awp) in self.policy_priors.iter().enumerate() {
            let score = awp.policy_score;

            let score_can_beat_current_best = match best_score {
                None => true,
                Some(curr) => {
                    score
                        .partial_cmp(&curr)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        == std::cmp::Ordering::Greater
                }
            };

            if !score_can_beat_current_best {
                continue;
            }

            let action_idx = idx as u32;
            let already_materialized = self.edges.iter().any(|e| e.action_idx() == action_idx);
            if already_materialized {
                continue;
            }

            best_action_idx = Some(action_idx);
            best_score = Some(score);
        }

        let Some(best_action_idx) = best_action_idx else {
            return;
        };

        self.edges.push(PUCTEdge::new(best_action_idx));
    }
}
