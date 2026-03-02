use common::TranspositionHash;
use dashmap::DashSet;
use model::GameAnalyzer;

type HashId = u64;

type AnalyzerState<M> = <M as GameAnalyzer>::State;

pub(super) struct AnalysisCoordinator<'a, M>
where
    M: GameAnalyzer,
{
    analyzer: &'a M,
    in_flight: DashSet<HashId>,
}

impl<'a, M> AnalysisCoordinator<'a, M>
where
    Self: Send + Sync,
    M: GameAnalyzer,
    AnalyzerState<M>: TranspositionHash,
{
    pub(super) fn new(analyzer: &'a M, capacity: usize) -> Self {
        Self {
            analyzer,
            in_flight: DashSet::with_capacity(capacity),
        }
    }

    pub(super) fn analyze(&self, game_state: AnalyzerState<M>) {
        let hash = game_state.transposition_hash();

        if self.in_flight.insert(hash) {
            self.analyzer.analyze(hash, &game_state);
        }
    }

    pub(super) fn complete(&self, hash: HashId) {
        self.in_flight.remove(&hash);
    }
}
