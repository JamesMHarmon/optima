use std::hash::Hash;
use fnv::FnvHashMap;
use super::analytics::{GameStateAnalysis};

pub struct AnalysisCache<S: Hash + Eq, A: Clone> {
    analysis_cache: FnvHashMap<S, GameStateAnalysis<A>>
}

impl<S: Hash + Eq, A: Clone> AnalysisCache<S, A> {
    pub fn new() -> Self {
        Self {
            analysis_cache: FnvHashMap::with_capacity_and_hasher(2_000_000, Default::default())
        }
    }

    pub fn get_or_insert<F: FnOnce() -> GameStateAnalysis<A>>(&mut self, game_state: S, default: F) -> GameStateAnalysis<A> {
        self.analysis_cache.entry(game_state).or_insert_with(default).clone()
    }
}
