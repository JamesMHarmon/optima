use anyhow::Result;
use crate::MCTSNode;

pub trait SelectionStrategy {
    type State;
    type Action;
    type Predictions;
    type PredicationValues;

    fn select_path(
        &self,
        node: &mut MCTSNode<Self::Action, Self::Predictions, Self::PredicationValues>,
        game_state: &Self::State,
        is_root: bool
    ) -> Result<usize>;
}
