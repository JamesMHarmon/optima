use std::mem;

use model::ActionWithPolicy;

use crate::edge::MCTSEdge;

#[derive(Debug)]
pub struct MCTSNode<A, P> {
    visits: usize,
    predictions: P,
    visited_edges: Vec<MCTSEdge<A, P>>,
    unvisited_edges: Vec<ActionWithPolicy<A>>,
}

impl<A, P> MCTSNode<A, P> {
    pub fn new(
        policy_scores: Vec<ActionWithPolicy<A>>,
        predictions: P,
    ) -> Self {
        Self {
            visits: 1,
            predictions,
            visited_edges: Vec::new(),
            unvisited_edges: policy_scores,
        }
    }

    pub fn get_node_visits(&self) -> usize {
        self.visits
    }

    pub fn get_child_by_index_mut(&mut self, index: usize) -> &mut MCTSEdge<A, F> {
        &mut self.visited_edges[index]
    }

    pub fn is_terminal(&self) -> bool {
        self.child_len() == 0
    }

    pub fn iter_visited_edges(&self) -> impl Iterator<Item = &MCTSEdge<A, F>> {
        self.visited_edges.iter()
    }

    pub fn iter_visited_edges_mut(&mut self) -> impl Iterator<Item = &mut MCTSEdge<A, F>> {
        self.visited_edges.iter_mut()
    }

    pub fn child_len(&self) -> usize {
        self.visited_edges.len() + self.unvisited_edges.len()
    }

    pub fn increment_visits(&mut self) {
        self.visits += 1;
    }

    pub fn visits(&self) -> usize {
        self.visits
    }

    pub fn set_visits(&mut self, visits: usize) {
        self.visits = visits;
    }

    pub fn predictions(&self) -> &P {
        &self.predictions
    }
}

impl<A, V, F> MCTSNode<A, V, F>
where
    A: Eq,
{
    pub fn get_position_of_visited_action(&self, action: &A) -> Option<usize> {
        self.iter_visited_edges().position(|c| c.action() == action)
    }
}

impl<A, V, F> MCTSNode<A, V, F>
where
    F: Default
{
    pub fn iter_visited_edges_and_top_unvisited_edge(
        &mut self,
    ) -> impl Iterator<Item = &MCTSEdge<A, F>> {
        self.init_top_policy_unvisited_action_if_all_initialized_edges_visited();
        self.visited_edges.iter()
    }

    pub fn iter_all_edges(&mut self) -> impl Iterator<Item = &mut MCTSEdge<A, F>> {
        self.init_all_edges();
        self.visited_edges.iter_mut()
    }

    fn init_all_edges(&mut self) {
        let unvisited_edges = mem::take(&mut self.unvisited_edges).into_iter();
        self.visited_edges.extend(unvisited_edges.map(Into::into));
    }

    /// Checks the set of initialized nodes to determine the next action to take.
    ///
    /// If there are unvisited nodes, no action is taken.
    /// If all nodes are visited, it selects the next valid action with the highest policy value.
    /// The selected action is then initialized as a new edge and added to the set of initialized nodes.
    fn init_top_policy_unvisited_action_if_all_initialized_edges_visited(&mut self) {
        if self.unvisited_edges.is_empty() {
            return;
        }

        let has_unvisited_edge = self.visited_edges.iter().any(|e| e.visits() == 0);

        if has_unvisited_edge {
            return;
        }

        let top_action_idx = self
            .unvisited_edges
            .iter()
            .enumerate()
            .max_by(|(_, e), (_, e2)| {
                e.policy_score
                    .partial_cmp(&e2.policy_score)
                    .unwrap_or_else(|| {
                        panic!(
                            "Could not compare two floats {} {}",
                            e.policy_score, e2.policy_score,
                        )
                    })
            })
            .map(|(index, _)| index)
            .expect("Should have found a top action idx.");

        let action_policy = self.unvisited_edges.swap_remove(top_action_idx);

        // Control the amount of capacity that is added to the vec as we don't want it doubling by default.
        if self.visited_edges.capacity() == self.visited_edges.len() {
            self.visited_edges.reserve(8);
        }

        self.visited_edges.push(action_policy.into());

        // Cleanup any spare capacity from the actions vec.
        if self.unvisited_edges.capacity() - self.unvisited_edges.len() > 16 {
            self.unvisited_edges.shrink_to_fit();
        }
    }
}


impl<A, V, F> MCTSNode<A, V, F>
where
    A: Eq,
    F: Default
{
    pub fn get_child_of_action(&mut self, action: &A) -> Option<&mut MCTSEdge<A, F>> {
        self.iter_all_edges().find(|e| e.action() == action)
    }
}

