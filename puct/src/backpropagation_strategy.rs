/// Strategy for backpropagating values through the PUCT search tree
///
/// Designed for concurrent access - the backpropagation strategy updates rollup statistics
/// that support atomic operations, allowing readers to access values while writes occur.
pub trait BackpropagationStrategy {
    type Predictions;
    type RollupStats;
    type NodeInfo;
    type State;

    /// Extract node-specific information from the game state.
    ///
    /// This is called during tree traversal to capture context needed for backpropagation,
    /// such as which player is to move at this node.
    fn node_info(&self, game_state: &Self::State) -> Self::NodeInfo;

    /// Backpropagate predictions through a single node.
    ///
    /// This method is called once for each node in the traversal path, from leaf to root.
    /// It should update the rollup statistics based on the predictions and node context.
    ///
    /// # Arguments
    /// * `node_info` - Context about the node (e.g., player to move)
    /// * `rollup_stats` - Statistics to update (e.g., value, game length, visit counts)
    /// * `predictions` - Terminal predictions from the leaf node
    fn backpropagate(
        &self,
        node_info: &Self::NodeInfo,
        rollup_stats: &Self::RollupStats,
        predictions: &Self::Predictions,
    );
}
