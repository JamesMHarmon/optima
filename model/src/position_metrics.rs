use super::node_metrics::NodeMetrics;

#[derive(Debug)]
pub struct PositionMetrics<S, A, P, PV> {
    pub game_state: S,
    pub node_metrics: NodeMetrics<A, P, PV>,
}
