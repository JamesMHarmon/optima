use model::NodeMetrics;
use serde::{Deserialize, Serialize};

type ActionAndMetrics<A, P, PV> = (A, NodeMetrics<A, P, PV>);

#[derive(Serialize, Deserialize, Debug)]
pub struct SelfPlayMetrics<A, P, PV> {
    analysis: Vec<ActionAndMetrics<A, P, PV>>,
    score: P,
}

impl<A, P, PV> SelfPlayMetrics<A, P, PV> {
    pub fn new(analysis: Vec<ActionAndMetrics<A, P, PV>>, score: P) -> Self {
        Self { analysis, score }
    }

    pub fn into_inner(self) -> (Vec<ActionAndMetrics<A, P, PV>>, P) {
        (self.analysis, self.score)
    }

    pub fn analysis(&self) -> &[ActionAndMetrics<A, P, PV>] {
        &self.analysis
    }

    pub fn score(&self) -> &P {
        &self.score
    }
}
