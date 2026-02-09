pub trait WeightedMerge {
    fn zero() -> Self;
    fn merge_weighted(&mut self, other: &Self, weight: u32);
}

pub trait RollupStats {
    type Snapshot: WeightedMerge;

    fn snapshot(&self) -> Self::Snapshot;

    fn set(&self, value: &Self::Snapshot);

    /// Merge `other` into `self` using explicit weights.
    /// - `self_weight`: the current visit/sample count represented by `self.snapshot()`.
    /// - `other_weight`: the visit/sample count represented by `other.snapshot()`.
    fn merge_rollup_weighted(&self, self_weight: u32, other: &Self, other_weight: u32) {
        let merged = Self::aggregate_weighted([
            (self.snapshot(), self_weight),
            (other.snapshot(), other_weight),
        ]);
        self.set(&merged);
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
