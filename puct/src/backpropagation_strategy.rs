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
        edges: &[EdgeInfo]
    );
}
