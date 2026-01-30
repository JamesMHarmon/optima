/// Strategy for backpropagating values through the PUCT search tree
/// 
/// Simplified design for single-writer architecture - no lending iterators needed
/// since the writer thread has exclusive access to the tree during backpropagation.
pub trait BackpropagationStrategy {
    type Action;
    type Predictions;
    type RollupStats;
    type State;

    /// Backpropagate predictions through a path of nodes
    /// 
    /// # Arguments
    /// * `path` - Slice of (node_id, edge_index) pairs representing the traversal path
    /// * `predictions` - Terminal predictions to backpropagate
    /// * `update_fn` - Callback to update each node/edge with computed values
    fn backpropagate(
        &self,
        node_visits: u32,
        rollup_stats: &mut Self::RollupStats,
        edges: &[EdgeInfo],
        _depth: usize,
    );
}

pub struct SimpleBackpropagationStrategy;

impl BackpropagationStrategy for SimpleBackpropagationStrategy {
    type Action = ();
    type Predictions = ();
    type RollupStats = ();
    type State = ();

    fn backpropagate(
        &self,
        node_visits: u32,
        rollup_stats: &mut Self::RollupStats,
        edges: &[EdgeInfo],
        _depth: usize,
    )
    {
                // Compute weighted average from all children
        let total_visits = node_visits as f32;
        let mut weighted_sum = 0.0;
        
        for edge in edges {
            let weight = edge.visits as f32 / total_visits;
            weighted_sum += edge.child_value * weight;
        }
        
        rollup_stats.value = weighted_sum;
    }
}
