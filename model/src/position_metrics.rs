use super::node_metrics::NodeMetrics;

#[derive(Debug)]
pub struct PositionMetrics<S, A> {
    pub game_state: S,
    pub score: f32,
    pub policy: NodeMetrics<A>
}
