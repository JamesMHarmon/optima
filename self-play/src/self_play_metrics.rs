use model::NodeMetrics;
use serde::{Deserialize, Serialize};

type ActionAndMetrics<A, P, SS> = (A, NodeMetrics<A, P, SS>);

#[derive(Serialize, Deserialize, Debug)]
pub struct SelfPlayMetrics<A, P, SS> {
    analysis: Vec<ActionAndMetrics<A, P, SS>>,
    terminal_score: P,
}

impl<A, P, SS> SelfPlayMetrics<A, P, SS> {
    pub fn new(analysis: Vec<ActionAndMetrics<A, P, SS>>, terminal_score: P) -> Self {
        Self {
            analysis,
            terminal_score,
        }
    }

    pub fn into_inner(self) -> (Vec<ActionAndMetrics<A, P, SS>>, P) {
        (self.analysis, self.terminal_score)
    }

    pub fn analysis(&self) -> &[ActionAndMetrics<A, P, SS>] {
        &self.analysis
    }

    pub fn terminal_score(&self) -> &P {
        &self.terminal_score
    }
}
