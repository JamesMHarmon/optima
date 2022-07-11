use model::NodeMetrics;
use serde::{Deserialize, Serialize};

type ActionAndMetrics<A, V> = (A, NodeMetrics<A, V>);

#[derive(Serialize, Deserialize, Debug)]
pub struct SelfPlayMetrics<A, V> {
    analysis: Vec<ActionAndMetrics<A, V>>,
    score: V,
}

impl<A, V> SelfPlayMetrics<A, V> {
    pub fn new(analysis: Vec<ActionAndMetrics<A, V>>, score: V) -> Self {
        Self { analysis, score }
    }

    pub fn into_inner(self) -> (Vec<ActionAndMetrics<A, V>>, V) {
        (self.analysis, self.score)
    }

    pub fn analysis(&self) -> &[ActionAndMetrics<A, V>] {
        &self.analysis
    }

    pub fn score(&self) -> &V {
        &self.score
    }
}
