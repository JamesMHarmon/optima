use model::position_metrics::PositionMetrics;
use model::{node_metrics::NodeMetrics, NodeChildMetrics};

use super::{Action, GameState, Value};

pub fn get_symmetries(
    metrics: PositionMetrics<GameState, Action, Value>,
) -> Vec<PositionMetrics<GameState, Action, Value>> {
    let PositionMetrics {
        game_state,
        policy,
        score,
        moves_left,
    } = &metrics;

    let symmetrical_state = game_state.get_vertical_symmetry();

    let symmetrical_metrics = PositionMetrics {
        game_state: symmetrical_state,
        policy: symmetrical_node_metrics(policy),
        score: score.clone(),
        moves_left: *moves_left,
    };

    vec![metrics, symmetrical_metrics]
}

fn symmetrical_node_metrics(metrics: &NodeMetrics<Action, Value>) -> NodeMetrics<Action, Value> {
    let children_symmetry = metrics
        .children
        .iter()
        .map(|m| NodeChildMetrics::new(m.action().invert_horizontal(), m.Q(), m.M(), m.visits()))
        .collect();

    NodeMetrics {
        visits: metrics.visits,
        value: metrics.value.clone(),
        moves_left: metrics.moves_left,
        children: children_symmetry,
    }
}
