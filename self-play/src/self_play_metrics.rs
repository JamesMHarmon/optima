use model::NodeMetrics;
use serde::{Deserialize, Serialize};

type ActionAndMetrics<A, P, PV> = (A, NodeMetrics<A, P, PV>);

#[derive(Serialize, Deserialize, Debug)]
pub struct SelfPlayMetrics<A, P, PV> {
    analysis: Vec<ActionAndMetrics<A, P, PV>>,
    terminal_score: P,
}

impl<A, P, PV> SelfPlayMetrics<A, P, PV> {
    pub fn new(analysis: Vec<ActionAndMetrics<A, P, PV>>, terminal_score: P) -> Self {
        Self {
            analysis,
            terminal_score,
        }
    }

    pub fn into_inner(self) -> (Vec<ActionAndMetrics<A, P, PV>>, P) {
        (self.analysis, self.terminal_score)
    }

    pub fn analysis(&self) -> &[ActionAndMetrics<A, P, PV>] {
        &self.analysis
    }

    pub fn terminal_score(&self) -> &P {
        &self.terminal_score
    }
}
