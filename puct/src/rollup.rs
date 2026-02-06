pub trait WeightedMerge {
    type Weight;

    fn zero() -> Self;
    fn merge_weighted(&mut self, other: &Self, weight: &Self::Weight);
}

pub trait RollupStats {
    type Snapshot: WeightedMerge;

    /// Update stats from a rollout result
    fn update(&self, value: &Self::Snapshot);

    /// Snapshot current stats
    fn snapshot(&self) -> Self::Snapshot;

    /// Aggregate weighted snapshots
    fn aggregate_weighted<'w, I>(iter: I) -> Self::Snapshot
    where
        I: IntoIterator<
            Item = (
                Self::Snapshot,
                &'w <Self::Snapshot as WeightedMerge>::Weight,
            ),
        >,
        <Self::Snapshot as WeightedMerge>::Weight: 'w,
    {
        let mut out = Self::Snapshot::zero();
        for (snap, weight) in iter {
            out.merge_weighted(&snap, weight);
        }
        out
    }
}

pub struct NodeContext {
    pub parent_visits: u32,
    pub depth: u16,
    pub is_root: bool,
}

pub trait PreparedEdgeScorer<R: RollupStats> {
    fn score(&self, prior: f32, child: &R::Snapshot) -> f64;
}

pub trait EdgeScorer<R: RollupStats> {
    type Prepared<'a>: PreparedEdgeScorer<R>
    where
        Self: 'a;

    fn prepare<'a>(&'a self, ctx: &NodeContext) -> Self::Prepared<'a>;
}

pub enum EdgeStats<R: RollupStats> {
    Direct {
        stats: R,
        prior: f32,
    },

    Afterstates {
        prior: f32,
        children: Vec<(R, <R::Snapshot as WeightedMerge>::Weight)>,
    },
}

pub fn edge_snapshot<R: RollupStats>(edge: &EdgeStats<R>) -> (f32, R::Snapshot) {
    match edge {
        EdgeStats::Direct { stats, prior } => (*prior, stats.snapshot()),

        EdgeStats::Afterstates { prior, children } => {
            let snap = R::aggregate_weighted(children.iter().map(|(r, w)| (r.snapshot(), w)));
            (*prior, snap)
        }
    }
}

pub fn edge_scores<'a, R, S, I>(
    edges: I,
    ctx: &NodeContext,
    scorer: &'a S,
) -> impl Iterator<Item = (usize, f64)> + 'a
where
    R: RollupStats + 'a,
    S: EdgeScorer<R>,
    I: IntoIterator<Item = &'a EdgeStats<R>> + 'a,
{
    let prepared = scorer.prepare(ctx);
    edges.into_iter().enumerate().map(move |(idx, edge)| {
        let (prior, snap) = edge_snapshot(edge);
        (idx, prepared.score(prior, &snap))
    })
}

pub fn select_best_edge<'a, R, S, I>(edges: I, ctx: &NodeContext, scorer: &'a S) -> usize
where
    R: RollupStats + 'a,
    S: EdgeScorer<R>,
    I: IntoIterator<Item = &'a EdgeStats<R>> + 'a,
{
    edge_scores(edges, ctx, scorer)
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}
