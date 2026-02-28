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

/// Per-edge scoring output for UI / debugging.
///
/// Note that `puct_score` is the *actual* score used by the selection policy
/// (including any policy-specific adjustments), not necessarily `Q + U`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct EdgeScore {
    pub edge_index: usize,
    pub usa: f32,
    pub cpuct: f32,
    pub puct_score: f32,
}

/// Optional extension trait that exposes per-edge scoring.
///
/// This is used to populate `EdgeDetails` with `Usa`, `cpuct`, and `puct_score`
/// for debugging/UGI output.
pub trait SelectionPolicyScoring<S>: SelectionPolicy<S> {
    fn score_edges<'a, I, A: 'a>(&self, node: NodeInfo, edges: I, state: &Self::State) -> Vec<EdgeScore>
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
