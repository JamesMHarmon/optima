use crate::MCTSNode;

pub struct SelectedNode<'node, N, A, P, PV> {
    pub node: &'node mut MCTSNode<A, P, PV>,
    pub selected_edge_index: usize,
    pub node_info: N,
}

pub trait NodeLendingIterator<'node, N, A, P, V> {
    fn next(&'node mut self) -> Option<SelectedNode<'node, N, A, P, V>>;
}

pub trait BackpropagationStrategy {
    type Action;
    type Predictions;
    type PropagatedValues;
    type NodeInfo;
    type State;

    fn node_info(&self, game_state: &Self::State) -> Self::NodeInfo;

    fn backpropagate<'node, I>(&'node self, visited_nodes: I, predictions: &Self::Predictions)
    where
        I: NodeLendingIterator<
            'node,
            Self::NodeInfo,
            Self::Action,
            Self::Predictions,
            Self::PropagatedValues,
        >;
}
