pub trait WeightedMerge {
    fn zero() -> Self;
    fn merge_weighted(&mut self, other: &Self, weight: u32);

    fn duplicate(&self) -> Self
    where
        Self: Sized,
    {
        let mut out = Self::zero();
        out.merge_weighted(self, 1);
        out
    }
}

pub trait RollupStats {
    type Snapshot: WeightedMerge;

    fn snapshot(&self) -> Self::Snapshot;

    fn set(&self, value: Self::Snapshot);

    fn accumulate(&self, sample: &Self::Snapshot) {
        let mut snap = self.snapshot();
        snap.merge_weighted(sample, 1);
        self.set(snap);
    }

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
