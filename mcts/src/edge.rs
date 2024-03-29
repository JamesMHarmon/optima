use crate::MCTSNodeState;
use generational_arena::Index;

#[allow(non_snake_case)]
#[derive(Debug)]
pub struct MCTSEdge<A> {
    pub action: A,
    pub W: f32,
    /// M is the expected length of the game. Needs to be divided by visits. THIS IS NOT MOVES LEFT!
    pub M: f32,
    pub visits: usize,
    pub policy_score: f32,
    pub node: MCTSNodeState,
}

impl<A> MCTSEdge<A> {
    pub fn node_index(&self) -> Option<Index> {
        self.node.get_index()
    }

    pub fn action(&self) -> &A {
        &self.action
    }
}
