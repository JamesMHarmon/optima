use model::NodeMetrics;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
pub struct SelfPlayMetrics<A, V> {
    analysis: Vec<(A, NodeMetrics<A>)>,
    score: V,
}

impl<A, V> SelfPlayMetrics<A, V> {
    pub fn new(analysis: Vec<(A, NodeMetrics<A>)>, score: V) -> Self {
        Self { analysis, score }
    }

    pub fn into_inner(self) -> (Vec<(A, NodeMetrics<A>)>, V) {
        (self.analysis, self.score)
    }

    pub fn analysis(&self) -> &[(A, NodeMetrics<A>)] {
        &self.analysis
    }

    pub fn score(&self) -> &V {
        &self.score
    }
}