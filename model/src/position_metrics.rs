use super::node_metrics::NodeMetrics;

#[derive(Debug)]
pub struct PositionMetrics<S, A, P, SS> {
    pub game_state: S,
    pub node_metrics: NodeMetrics<A, P, SS>,
}
