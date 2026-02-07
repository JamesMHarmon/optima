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

/// Statistics for an edge, either direct or aggregated from afterstates
pub enum EdgeStats<'a, R: RollupStats> {
    /// Direct edge with single rollup stats
    Direct { stats: &'a R },

    /// Edge with multiple afterstate outcomes to aggregate
    /// Each outcome has rollup stats and a weight (typically visit count)
    Afterstates { outcomes: &'a [(R, u32)] },
}

impl<'a, R: RollupStats> EdgeStats<'a, R> {
    /// Compute the aggregated snapshot for this edge
    pub fn snapshot(&self) -> R::Snapshot {
        match self {
            EdgeStats::Direct { stats } => stats.snapshot(),
            EdgeStats::Afterstates { outcomes } => {
                R::aggregate_weighted(outcomes.iter().map(|(r, w)| (r.snapshot(), *w)))
            }
        }
    }
}
