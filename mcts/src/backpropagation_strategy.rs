use generational_arena::Index;

use crate::MCTSNode;

pub trait BackpropagationStrategy {
    type Action;
    type Predictions;
    type PredicationValues;
    type NodeInfo;
    type State;

    fn node_info(&self, game_state: &Self::State) -> Self::NodeInfo;

    fn backpropagate(
        &self,
        visited_nodes: &[NodeUpdateInfo],
        evaluated_node_index: Index,
        evaluated_node_move_num: usize,
        arena: &mut NodeArenaInner<MCTSNode<Self::Action, Self::Predictions, Self::PredicationValues>>,
    );
}
