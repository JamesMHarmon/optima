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
        let mut snap = Self::Snapshot::zero();
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
