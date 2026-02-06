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
/// predictions into canonical statistics. Aggregation is handled by the RollupStats
/// trait methods.
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
}
