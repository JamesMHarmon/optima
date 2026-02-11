use crate::{EdgeInfo, RollupStats, SelectionPolicy};

pub trait ScorerSelectionPolicy<R: RollupStats>: EdgeScorer<R> {
    fn select_with_scorer<'a, I, A: 'a>(&self, edges: I, node_visits: u32, depth: u16) -> usize
    where
        I: Iterator<Item = EdgeInfo<'a, A, R::Snapshot>>,
        R::Snapshot: 'a,
    {
        let ctx = NodeContext { node_visits, depth };

        let prepared = self.prepare(&ctx);

        edges
            .enumerate()
            .map(|(idx, e)| {
                let score = prepared.score(e.policy_prior, e.snapshot.as_ref());
                (idx, score)
            })
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    }
}

pub struct NodeContext {
    pub node_visits: u32,
    pub depth: u16,
}

pub trait PreparedEdgeScorer<R: RollupStats> {
    fn score(&self, prior: f32, child: Option<&R::Snapshot>) -> f64;
}

pub trait EdgeScorer<R: RollupStats> {
    type Prepared<'a>: PreparedEdgeScorer<R>
    where
        Self: 'a;

    fn prepare<'a>(&'a self, ctx: &NodeContext) -> Self::Prepared<'a>;
}

impl<T, R> ScorerSelectionPolicy<R> for T
where
    T: EdgeScorer<R>,
    R: RollupStats,
{
}

pub struct CpuctPolicy<S> {
    pub scorer: S,
}

impl<S> CpuctPolicy<S> {
    pub fn new(scorer: S) -> Self {
        Self { scorer }
    }
}

impl<R, S> SelectionPolicy<R> for CpuctPolicy<S>
where
    R: RollupStats,
    S: EdgeScorer<R>,
{
    type State = ();

    fn select_edge<'a, I, A: 'a>(
        &self,
        edges: I,
        node_visits: u32,
        _state: &Self::State,
        depth: u16,
    ) -> usize
    where
        I: Iterator<Item = EdgeInfo<'a, A, R::Snapshot>>,
        R::Snapshot: 'a,
    {
        self.scorer.select_with_scorer(edges, node_visits, depth)
    }
}
