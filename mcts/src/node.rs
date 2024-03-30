use model::ActionWithPolicy;

use crate::edge::MCTSEdge;

#[derive(Debug)]
pub struct MCTSNode<A, V> {
    visits: usize,
    value_score: V,
    moves_left_score: f32,
    // unvisited_edges: Vec<ActionWithPolicy<A>>,
    visited_edges: Vec<MCTSEdge<A>>,
}

impl<A, V> MCTSNode<A, V> {
    pub fn new(
        policy_scores: Vec<ActionWithPolicy<A>>,
        value_score: V,
        moves_left_score: f32,
    ) -> Self {
        Self {
            visits: 1,
            value_score,
            moves_left_score,
            // unvisited_edges: policy_scores,
            visited_edges: policy_scores.into_iter().map(|p| p.into()).collect(),
        }
    }

    pub fn get_node_visits(&self) -> usize {
        self.visits
    }

    pub fn get_child_by_index_mut(&mut self, index: usize) -> &mut MCTSEdge<A> {
        &mut self.visited_edges[index]
    }

    pub fn is_terminal(&self) -> bool {
        self.child_len() == 0
    }

    pub fn iter_all_edges(&self) -> impl Iterator<Item = &MCTSEdge<A>> {
        // @TODO
        self.visited_edges.iter()
    }

    pub fn iter_visited_edges(&self) -> impl Iterator<Item = &MCTSEdge<A>> {
        // @TODO
        self.visited_edges.iter()
    }

    pub fn iter_visited_edges_and_top_unvisited_edge(&self) -> impl Iterator<Item = &MCTSEdge<A>> {
        // @TODO
        self.visited_edges.iter()
    }

    pub fn child_len(&self) -> usize {
        // @TODO
        self.visited_edges.len()
        // self.visited_edges.is_empty() // && self.unvisited_edges.is_empty()
    }

    pub fn iter_edges_mut(&mut self) -> impl Iterator<Item = &mut MCTSEdge<A>> {
        self.visited_edges.iter_mut()
    }

    pub fn init_all_edges(&mut self) {
        // @TODO
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

    pub fn value_score(&self) -> &V {
        &self.value_score
    }

    pub fn moves_left_score(&self) -> f32 {
        self.moves_left_score
    }
}

impl<A, V> MCTSNode<A, V>
where
    A: Eq,
{
    pub fn get_child_of_action(&self, action: &A) -> Option<&MCTSEdge<A>> {
        // @TODO
        self.iter_all_edges().find(|c| c.action() == action)
    }

    pub fn get_position_of_action(&self, action: &A) -> Option<usize> {
        // @TODO
        self.iter_all_edges().position(|c| c.action() == action)
    }
}
