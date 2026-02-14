use crate::{EdgeInfo, SelectionPolicy};

type ScSnapshot<Sc> = <Sc as EdgeScorer>::Snapshot;

pub trait ScorerSelectionPolicy: EdgeScorer {
    fn select_with_scorer<'a, I, A: 'a>(&self, edges: I, node_visits: u32, depth: u32) -> usize
    where
        I: Iterator<Item = EdgeInfo<'a, A, Self::Snapshot>>,
        Self::Snapshot: 'a,
    {
        let ctx = NodeContext { node_visits, depth };

        let prepared = self.prepare(&ctx);

        edges
            .map(|e| {
                let score = prepared.score(e.policy_prior, e.snapshot.as_ref());
                (e.edge_index, score)
            })
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(edge_index, _)| edge_index)
            .expect("Expected at least one edge for selection")
    }
}

pub struct NodeContext {
    pub node_visits: u32,
    pub depth: u32,
}

pub trait PreparedEdgeScorer<S> {
    fn score(&self, prior: f32, child: Option<&S>) -> f64;
}

pub trait EdgeScorer {
    type Snapshot;

    type Prepared<'a>: PreparedEdgeScorer<Self::Snapshot>
    where
        Self: 'a;

    fn prepare<'a>(&'a self, ctx: &NodeContext) -> Self::Prepared<'a>;
}

impl<T> ScorerSelectionPolicy for T where T: EdgeScorer {}

pub struct CpuctPolicy<Sc> {
    pub scorer: Sc,
}

impl<Sc> CpuctPolicy<Sc> {
    pub fn new(scorer: Sc) -> Self {
        Self { scorer }
    }
}

impl<Sc> SelectionPolicy<ScSnapshot<Sc>> for CpuctPolicy<Sc>
where
    Sc: EdgeScorer,
{
    type State = ();

    fn select_edge<'a, I, A: 'a>(
        &self,
        edges: I,
        node_visits: u32,
        _state: &Self::State,
        depth: u32,
    ) -> usize
    where
        I: Iterator<Item = EdgeInfo<'a, A, ScSnapshot<Sc>>>,
        ScSnapshot<Sc>: 'a,
    {
        self.scorer.select_with_scorer(edges, node_visits, depth)
    }
}
