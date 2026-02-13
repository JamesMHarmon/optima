use super::RollupStats;

/// Defines the value semantics of the search as well as how game outcomes and
/// neural network predictions are converted into a canonical `Snapshot`.
///
/// Snapshots **must be canonical** to support graph-based search.
/// The same position may be reached from different paths
/// with different players to move, so snapshots must have a consistent
/// interpretation independent of perspective.
///
/// Examples of canonical representations:
///
/// - Store values for both players: `(player_0_value, player_1_value)`
/// - Store WDL probabilities: `(win_prob, draw_prob, loss_prob)`
/// - Store values from a fixed global perspective (e.g., always Player 0)
///
/// The search engine will aggregate snapshots algebraically using
/// `RollupStats`. `ValueModel` does not define how aggregation
/// works, only how predictions become canonical snapshots.
pub trait ValueModel {
    type State;
    type Predictions;
    type Terminal;
    type Rollup: RollupStats + From<<Self::Rollup as RollupStats>::Snapshot>;

    fn pred_snapshot(
        &self,
        state: &Self::State,
        predictions: &Self::Predictions,
    ) -> <Self::Rollup as RollupStats>::Snapshot;

    fn terminal_snapshot(
        &self,
        state: &Self::State,
        terminal: &Self::Terminal,
    ) -> <Self::Rollup as RollupStats>::Snapshot;
}
