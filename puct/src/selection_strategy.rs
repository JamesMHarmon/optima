use crate::{EdgeScorer, EdgeStats, NodeContext, PreparedEdgeScorer, RollupStats};

/// Policy for selecting edges during tree traversal
///
/// Used by read-only selection workers to compute PUCT values and choose edges.
/// Implementations define the exploration formula (standard PUCT, AlphaZero variant, etc.)
pub trait SelectionPolicy<R: RollupStats> {
    type State;

    /// Select which edge to follow from the current node
    ///
    /// # Arguments
    /// * `edges` - Iterator of edges to choose from
    /// * `node_visits` - Total visits to the current node (parent_visits in EdgeScorer context)
    /// * `state` - Current game state
    /// * `depth` - Depth in tree (0 = root, increases toward leaves)
    ///
    /// # Returns
    /// Index of the selected edge
    fn select_edge<'a, I, A: 'a>(
        &self,
        edges: I,
        node_visits: u32,
        state: &Self::State,
        depth: u16,
    ) -> usize
    where
        I: Iterator<Item = EdgeInfo<'a, A, R>>,
        R: 'a;
}

/// Read-only information about an edge for selection
///
/// Provides all information needed to compute PUCT scores without mutation.
/// Supports both direct edges and afterstate edges with multiple outcomes.
pub struct EdgeInfo<'a, A, R: RollupStats> {
    pub action: &'a A,
    pub policy_prior: f32,
    pub visits: u32,
    /// Edge statistics - None for unvisited edges
    pub stats: Option<EdgeStats<'a, R>>,
}

/// Helper to implement SelectionPolicy using an EdgeScorer
///
/// This provides a default selection implementation that:
/// 1. Uses the provided EdgeScorer to compute scores for visited edges
/// 2. Properly aggregates afterstate outcomes when present
/// 3. Handles unvisited edges (visits=0) with high exploration priority
/// 4. Selects the edge with the highest score
///
/// Note: Unvisited edges are given infinite score to prioritize exploration.
/// For visited edges, the EdgeStats snapshot is computed (aggregating afterstates
/// if present) and passed to the scorer.
///
/// Use this when your selection policy can be expressed as an EdgeScorer.
pub fn select_edge_with_scorer<'a, R, S, I, A>(
    edges: I,
    scorer: &S,
    parent_visits: u32,
    depth: u16,
    is_root: bool,
) -> usize
where
    R: RollupStats + 'a,
    S: EdgeScorer<R>,
    I: Iterator<Item = EdgeInfo<'a, A, R>>,
    A: 'a,
{
    let ctx = NodeContext {
        parent_visits,
        depth,
        is_root,
    };

    let prepared = scorer.prepare(&ctx);

    // Score all edges
    let (best_idx, _): (usize, f64) = edges
        .enumerate()
        .map(|(idx, e)| {
            let score = match &e.stats {
                None => {
                    // Unvisited edge: prioritize exploration
                    f64::INFINITY
                }
                Some(edge_stats) => {
                    // Visited edge: compute snapshot (handles both direct and afterstate cases)
                    let snap = edge_stats.snapshot();
                    PreparedEdgeScorer::<R>::score(&prepared, e.policy_prior, &snap)
                }
            };
            (idx, score)
        })
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or((0, 0.0));

    best_idx
}
