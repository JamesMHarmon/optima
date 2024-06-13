use anyhow::Result;
use crate::{EdgeDetails, MCTSNode};

pub trait SelectionStrategy {
    type State;
    type Action;
    type Predictions;
    type PropagatedValues;

    fn select_path(
        &self,
        node: &mut MCTSNode<Self::Action, Self::Predictions, Self::PropagatedValues>,
        game_state: &Self::State,
        is_root: bool
    ) -> Result<usize>;

    fn node_details(
        &self,
        node: &mut MCTSNode<Self::Action, Self::Predictions, Self::PropagatedValues>,
        game_state: &Self::State,
        is_root: bool
    ) -> Vec<EdgeDetails<Self::Action, Self::PropagatedValues>>;
}
