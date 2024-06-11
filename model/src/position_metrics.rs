use super::node_metrics::NodeMetrics;

#[derive(Debug)]
pub struct PositionMetrics<S, A, P, PV> {
    pub game_state: S,
    /// Final game score known as Z. May have Q mixed through Q mixing or deblundering.
    pub policy: NodeMetrics<A, P, PV>,
}
