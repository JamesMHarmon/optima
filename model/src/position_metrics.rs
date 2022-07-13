use super::node_metrics::NodeMetrics;

#[derive(Debug)]
pub struct PositionMetrics<S, A, V> {
    pub game_state: S,
    /// Final game score known as Z. May have Q mixed through Q mixing or deblundering.
    pub score: V,
    pub policy: NodeMetrics<A, V>,
    /// The number of moves left calculated by taking the last move number of the game and subtracting the move number of this state plus one.
    pub moves_left: usize,
}
