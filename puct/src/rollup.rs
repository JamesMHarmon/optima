pub trait WeightedMerge {
    fn zero() -> Self;
    fn merge_weighted(&mut self, other: &Self, weight: u32);
}

pub trait RollupStats {
    type Snapshot: WeightedMerge;

    /// Update stats from a rollout result
    fn update(&self, value: &Self::Snapshot);

    /// Snapshot current stats
    fn snapshot(&self) -> Self::Snapshot;

    /// Merge another RollupStats into this one with a weight
    fn merge_rollup_weighted(&self, other: &Self, weight: u32) {
        let mut snap = self.snapshot();
        snap.merge_weighted(&other.snapshot(), weight);
        self.update(&snap);
    }

    /// Aggregate weighted snapshots
    fn aggregate_weighted<I>(iter: I) -> Self::Snapshot
    where
        I: IntoIterator<Item = (Self::Snapshot, u32)>,
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

pub enum EdgeStats<'a, R: RollupStats> {
    Direct {
        stats: &'a R,
        prior: f32,
    },

    Afterstates {
        prior: f32,
        outcomes: &'a [(R, u32)],
    },
}

pub fn edge_snapshot<R: RollupStats>(edge: &EdgeStats<'_, R>) -> (f32, R::Snapshot) {
    match edge {
        EdgeStats::Direct { stats, prior } => (*prior, stats.snapshot()),

        EdgeStats::Afterstates { prior, outcomes } => {
            let snap = R::aggregate_weighted(outcomes.iter().map(|(r, w)| (r.snapshot(), *w)));
            (*prior, snap)
        }
    }
}

pub fn edge_scores<'a, R, S, I>(
    edges: I,
    scorer: &'a S,
    ctx: &NodeContext,
) -> impl Iterator<Item = (usize, f64)> + 'a
where
    R: RollupStats + 'a,
    S: EdgeScorer<R>,
    I: IntoIterator<Item = &'a EdgeStats<'a, R>> + 'a,
{
    let prepared = scorer.prepare(ctx);
    edges.into_iter().enumerate().map(move |(idx, edge)| {
        let (prior, snap) = edge_snapshot(edge);
        (idx, prepared.score(prior, &snap))
    })
}

pub fn select_best_edge<'a, R, S, I>(edges: I, scorer: &'a S, ctx: &NodeContext) -> usize
where
    R: RollupStats + 'a,
    S: EdgeScorer<R>,
    I: IntoIterator<Item = &'a EdgeStats<'a, R>> + 'a,
{
    edge_scores(edges, scorer, ctx)
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}
