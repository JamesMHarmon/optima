/// Policy for selecting edges during tree traversal
///
/// Used by read-only selection workers to compute PUCT values and choose edges.
/// Implementations define the exploration formula (standard PUCT, AlphaZero variant, etc.)
pub trait SelectionPolicy {
    type State;

    /// Select which edge to follow from the current node
    ///
    /// # Arguments
    /// * `edges` - Iterator of edges to choose from
    /// * `node_visits` - Total visits to the current node
    /// * `state` - Current game state
    /// * `depth` - Depth in tree (0 = root, increases toward leaves)
    ///
    /// # Returns
    /// Index of the selected edge
    fn select_edge<'a, I, A: 'a, R: 'a>(
        &self,
        edges: I,
        node_visits: u32,
        state: &Self::State,
        depth: usize,
    ) -> usize
    where
        I: Iterator<Item = EdgeInfo<'a, A, R>>;
}

/// Read-only information about an edge for selection
///
/// Provides all information needed to compute PUCT scores without mutation.
pub struct EdgeInfo<'a, A, R> {
    pub action: &'a A,
    pub policy_prior: f32,
    pub visits: u32,
    pub rollup_stats: Option<&'a R>,
}
