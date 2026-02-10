/// Terminal node representing a final game state.
pub struct Terminal<R> {
    rollup_stats: R,
}

impl<R> Terminal<R> {
    pub fn new(rollup_stats: R) -> Self {
        Self { rollup_stats }
    }

    pub fn rollup_stats(&self) -> &R {
        &self.rollup_stats
    }
}
