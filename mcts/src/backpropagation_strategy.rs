use crate::MCTSNode;

type InfoNodePair<'node, N, A, P, PV> = (N, &'node mut MCTSNode<A, P, PV>);

pub trait NodeLendingIterator<'node, N, A, P, V> {
    fn next(&'node mut self) -> Option<InfoNodePair<'node, N, A, P, V>>;
}

pub trait BackpropagationStrategy {
    type Action;
    type Predictions;
    type PropagatedValues;
    type NodeInfo;
    type State;

    fn node_info(&self, game_state: &Self::State) -> Self::NodeInfo;

    fn backpropagate<'node, I>(
        &self,
        visited_nodes: I,
        predictions: &Self::Predictions,
    ) where
        I: NodeLendingIterator<'node, Self::NodeInfo, Self::Action, Self::Predictions, Self::PropagatedValues>,
        Self::Action: 'node,
        Self::Predictions: 'node,
        Self::PropagatedValues: 'node;
}
