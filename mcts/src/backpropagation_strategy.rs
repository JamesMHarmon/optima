use crate::MCTSNode;

pub trait BackpropagationStrategy {
    type Action;
    type Predictions;
    type PredicationValues;
    type NodeInfo;
    type State;

    fn node_info(&self, game_state: &Self::State) -> Self::NodeInfo;

    fn backpropagate<'a, I>(
        &self,
        visited_nodes: I,
        evaluated_node: &'a mut MCTSNode<Self::Action, Self::Predictions, Self::PredicationValues>,
    ) where
        I: Iterator<
            Item = (
                Self::NodeInfo,
                &'a mut MCTSNode<Self::Action, Self::Predictions, Self::PredicationValues>,
            ),
        >,
        Self::Action: 'a,
        Self::Predictions: 'a,
        Self::PredicationValues: 'a;
}
