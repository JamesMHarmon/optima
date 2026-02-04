/// Strategy for backpropagating values through the PUCT search tree
///
/// Designed for concurrent access - the backpropagation strategy updates rollup statistics
/// that support atomic operations, allowing readers to access values while writes occur.
///
/// The strategy handles three types of nodes:
/// - StateNode: has rollup_stats that aggregate values from child edges
/// - AfterStateNode: rollup_stats computed dynamically from outcomes (not stored)
/// - TerminalNode: has rollup_stats from terminal evaluation
pub trait BackpropagationStrategy {
    type Predictions;
    type RollupStats;
    type StateInfo;
    type State;

    /// Extract state-specific information from the game state.
    ///
    /// This is called during node creation to capture context needed for backpropagation,
    /// such as which player is to move at this state.
    fn state_info(&self, game_state: &Self::State) -> Self::StateInfo;

    /// Create initial rollup statistics from predictions.
    ///
    /// Called when creating a new State or Terminal node to initialize its statistics.
    ///
    /// # Arguments
    /// * `state_info` - Context about the state (e.g., player to move)
    /// * `predictions` - Predictions to initialize stats from
    fn create_rollup_stats(
        &self,
        state_info: &Self::StateInfo,
        predictions: &Self::Predictions,
    ) -> Self::RollupStats;

    /// Update a StateNode's rollup statistics during backpropagation.
    ///
    /// This method is called once for each StateNode in the traversal path, from leaf to root.
    /// It should aggregate values from the node's child edges using weighted averaging.
    ///
    /// # Arguments
    /// * `rollup_stats` - Statistics to update (e.g., value, game length, visit counts)
    /// * `children` - Iterator over (child_rollup_stats, visits) pairs for weighted averaging
    fn update_state_stats<'a, I>(
        &self,
        rollup_stats: &Self::RollupStats,
        children: I,
    ) where
        I: Iterator<Item = (&'a Self::RollupStats, u32)>,
        Self::RollupStats: 'a;

    /// Aggregate AfterStateNode rollup statistics from its outcomes.
    ///
    /// AfterStateNodes don't store rollup_stats - instead they are computed dynamically
    /// by aggregating over the outcomes (stochastic transitions).
    ///
    /// # Arguments
    /// * `outcomes` - Iterator over (outcome_rollup_stats, visits) pairs
    ///
    /// # Returns
    /// Aggregated rollup statistics, or None if no outcomes exist yet
    fn aggregate_after_state_stats<'a, I>(
        &self,
        outcomes: I,
    ) -> Option<Self::RollupStats>
    where
        I: Iterator<Item = (&'a Self::RollupStats, u32)>,
        Self::RollupStats: 'a;
}
