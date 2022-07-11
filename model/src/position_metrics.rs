use super::node_metrics::NodeMetrics;

#[derive(Debug)]
pub struct PositionMetrics<S, A, V> {
    pub game_state: S,
    pub score: V,
    pub policy: NodeMetrics<A, V>,
    pub moves_left: usize,
}
