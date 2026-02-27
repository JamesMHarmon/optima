/// Policy for selecting edges during tree traversal
///
/// Used by read-only selection workers to compute PUCT values and choose edges.
/// Implementations define the exploration formula (standard PUCT, AlphaZero variant, etc.)
pub trait SelectionPolicy<S> {
    type State;

    fn select_edge<'a, I, A: 'a>(&self, node: NodeInfo, edges: I, state: &Self::State) -> usize
    where
        I: Iterator<Item = EdgeInfo<'a, A, S>>,
        S: 'a;
}

/// Read-only information about the current node for selection.
#[derive(Clone, Copy, Debug, Default)]
pub struct NodeInfo {
    pub visits: u32,
    pub virtual_visits: u32,
    pub depth: u32,
}

impl NodeInfo {
    #[inline]
    pub fn total_visits(self) -> u32 {
        self.visits + self.virtual_visits
    }

    #[inline]
    pub fn is_root(self) -> bool {
        self.depth == 0
    }
}

/// Read-only information about an edge for selection
///
/// Provides all information needed to compute PUCT scores without mutation.
/// Supports both direct edges and afterstate edges with multiple outcomes.
pub struct EdgeInfo<'a, A, S> {
    pub edge_index: usize,
    pub action: &'a A,
    pub policy_prior: f32,
    pub visits: u32,
    pub virtual_visits: u32,
    pub snapshot: Option<S>,
}
