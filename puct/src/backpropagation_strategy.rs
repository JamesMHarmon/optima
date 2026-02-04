/// Strategy for backpropagating values through the PUCT search tree
///
/// Designed for concurrent access - the backpropagation strategy updates rollup statistics
/// that support atomic operations, allowing readers to access values while writes occur.
///
/// The strategy handles three types of nodes:
/// - StateNode: has rollup_stats that aggregate values from child edges
/// - AfterStateNode: rollup_stats computed dynamically from outcomes (not stored)
/// - TerminalNode: has rollup_stats from terminal evaluation
///
/// ## RollupStats Requirements
///
/// **RollupStats MUST be canonical (perspective-agnostic)** to work correctly with
/// transposition tables in graph-based search. Because the same position can be reached
/// from different paths with different players to move, rollup statistics must have a
/// consistent interpretation regardless of whose turn it is.
///
/// Examples of canonical representations:
/// - Store values for both players: `(player_0_value, player_1_value)`
/// - Store WDL probabilities: `(win_prob, draw_prob, loss_prob)`
/// - Store from a fixed perspective: always Player 0's viewpoint
///
/// The `create_rollup_stats` method uses `StateInfo` to convert player-relative
/// predictions into canonical statistics. The `aggregate_stats` method assumes
/// all statistics are already canonical and can be aggregated directly.
pub trait BackpropagationStrategy {
    type Predictions;
    type RollupStats;
    type StateInfo;
    type State;

    /// Extract state-specific information from the game state.
    ///
    /// This is called during node creation to capture context needed for converting
    /// predictions into canonical (perspective-agnostic) rollup statistics.
    /// For example, knowing which player is to move allows converting player-relative
    /// predictions (e.g., "win probability for current player") into canonical form
    /// (e.g., "win probability for player 0").
    fn state_info(&self, game_state: &Self::State) -> Self::StateInfo;

    /// Create initial rollup statistics from predictions.
    ///
    /// Called when creating a new State or Terminal node to initialize its statistics.
    /// Must convert predictions into canonical (perspective-agnostic) form using the
    /// state_info context.
    ///
    /// # Arguments
    /// * `state_info` - Context about the state (e.g., player to move)
    /// * `predictions` - Predictions to initialize stats from (may be player-relative)
    ///
    /// # Returns
    /// Canonical rollup statistics that can be aggregated with stats from any other node
    fn create_rollup_stats(
        &self,
        state_info: &Self::StateInfo,
        predictions: &Self::Predictions,
    ) -> Self::RollupStats;

    /// Aggregate rollup statistics using weighted averaging.
    ///
    /// Takes an iterator of (rollup_stats, weight) pairs and computes the weighted average,
    /// writing the result to the target rollup stats. All input statistics must already be
    /// in canonical (perspective-agnostic) form, allowing direct aggregation without
    /// perspective conversion.
    ///
    /// Used for both:
    /// - StateNode backpropagation: aggregate values from child edges
    /// - AfterState value computation: aggregate values from stochastic outcomes
    ///
    /// # Arguments
    /// * `target` - RollupStats to write the aggregated result to (uses atomic operations)
    /// * `weighted_stats` - Iterator over (canonical_rollup_stats, weight) pairs where weight is typically visit count
    fn aggregate_stats<'a, I>(
        &self,
        target: &Self::RollupStats,
        weighted_stats: I,
    ) where
        I: Iterator<Item = (&'a Self::RollupStats, u32)>,
        Self::RollupStats: 'a;
}
