/// Policy for selecting edges during tree traversal.
///
/// `SS` is the rollup snapshot type produced by the value model.
/// `type State` is the game state used by `select_edge`.
///
use std::collections::HashSet;
/// Note that `select_edge` keeps its own free generic `A` (not related to any action type).
/// This is intentional: scoring only needs edge metadata (visits, prior,
/// snapshot), so the same policy struct can be reused for every game without
/// being re-parameterised on the concrete action type.
pub trait SelectionPolicy<SS> {
    type State;
    type Action;
    type Terminal;

    /// Choose an edge index to follow during tree traversal.
    fn select_edge<'a, I>(
        &self,
        node: NodeInfo<SS>,
        edges: I,
        state: &Self::State,
        depth: u32,
    ) -> usize
    where
        I: Iterator<Item = EdgeInfo<'a, Self::Action, SS>>,
        Self::Action: 'a,
        SS: 'a;

    /// Called during tree traversal after a state transition to detect
    /// path-context-dependent terminals that the transposition table cannot represent.
    ///
    /// `path_hashes` contains the transposition hash of every game state visited
    /// so far on the current simulation path (root first, newest last).
    ///
    /// Default implementation returns `None`.
    fn terminal_for_trajectory(
        &self,
        state: &Self::State,
        visited: &HashSet<u64>,
    ) -> Option<Self::Terminal>;
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
pub trait SelectionPolicyScoring<SS>: SelectionPolicy<SS> {
    fn score_edges<'a, I>(
        &self,
        node: NodeInfo<SS>,
        edges: I,
        state: &Self::State,
        depth: u32,
    ) -> Vec<EdgeScore>
    where
        I: Iterator<Item = EdgeInfo<'a, Self::Action, SS>>,
        Self::Action: 'a,
        SS: 'a;
}

/// Read-only information about the current node for selection.
#[derive(Clone, Copy, Debug)]
pub struct NodeInfo<SS> {
    pub visits: u32,
    pub virtual_visits: u32,
    pub snapshot: SS,
}

impl<SS> NodeInfo<SS> {
    pub fn new(visits: u32, virtual_visits: u32, snapshot: SS) -> Self {
        Self {
            visits,
            virtual_visits,
            snapshot,
        }
    }

    #[inline]
    pub fn total_visits(self) -> u32 {
        self.visits + self.virtual_visits
    }
}

/// Read-only information about an edge for selection
///
/// Provides all information needed to compute PUCT scores without mutation.
/// Supports both direct edges and afterstate edges with multiple outcomes.
pub struct EdgeInfo<'a, A, SS> {
    pub edge_index: usize,
    pub action: &'a A,
    pub policy_prior: f32,
    pub visits: u32,
    pub virtual_visits: u32,
    pub snapshot: Option<SS>,
}

/// Called during tree traversal *before* `take_action` is applied to detect
/// path-context-dependent terminals that the transposition table cannot represent.
///
/// Returns `Some(terminal)` to short-circuit traversal when the action leads to a
/// terminal that depends on the path taken to reach the current node — e.g. a
/// 3rd-position repetition in Arimaa (loss for the repeater) or
/// draw-by-repetition in Quoridor.
///
/// `visited` is the set of transposition hashes of every game state visited
/// so far on the current simulation path.
///
/// Implement on a named struct and inject it into the selection policy as a
/// dependency (see `NoTrajectoryTerminal` for the no-op default).
pub trait TrajectoryTerminal {
    type State;
    type Terminal;

    fn terminal_for_trajectory(
        &self,
        state: &Self::State,
        visited: &HashSet<u64>,
    ) -> Option<Self::Terminal>;
}

/// No-op implementation of [`TrajectoryTerminal`] for policies that do not need
/// path-context-dependent terminal detection.
///
/// The `St` and `Term` type parameters are inferred from the usage context
/// (e.g. the game's state and terminal types), so callers typically just
/// pass `NoTrajectoryTerminal::default()`.
pub struct NoTrajectoryTerminal<St, Term>(std::marker::PhantomData<fn(St) -> Term>);

impl<St, Term> Default for NoTrajectoryTerminal<St, Term> {
    fn default() -> Self {
        Self(std::marker::PhantomData)
    }
}

impl<St, Term> TrajectoryTerminal for NoTrajectoryTerminal<St, Term> {
    type State = St;
    type Terminal = Term;

    fn terminal_for_trajectory(
        &self,
        _state: &Self::State,
        _visited: &HashSet<u64>,
    ) -> Option<Term> {
        None
    }
}
