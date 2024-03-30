use std::rc::Rc;

use futures_intrusive::sync::LocalManualResetEvent;
use generational_arena::Index;
use model::ActionWithPolicy;

#[allow(non_snake_case)]
#[derive(Debug)]
pub struct MCTSEdge<A> {
    action: A,
    W: f32,
    M: f32,
    visits: usize,
    policy_score: f32,
    node: MCTSNodeState,
}

#[allow(non_snake_case)]
impl<A> MCTSEdge<A> {
    pub fn new(action: A, policy_score: f32) -> Self {
        Self {
            action,
            policy_score,
            W: 0.0,
            M: 0.0,
            visits: 0,
            node: MCTSNodeState::Unexpanded,
        }
    }

    pub fn node_index(&self) -> Option<Index> {
        self.node.get_index()
    }

    pub fn action(&self) -> &A {
        &self.action
    }

    pub fn policy_score(&self) -> f32 {
        self.policy_score
    }

    pub fn set_policy_score(&mut self, policy_score: f32) {
        self.policy_score = policy_score
    }

    pub fn visits(&self) -> usize {
        self.visits
    }

    pub fn set_visits(&mut self, visits: usize) {
        self.visits = visits;
    }

    pub fn increment_visits(&mut self) {
        self.visits += 1;
    }

    /// M is the sum of the expected length of the game of all child nodes. Needs to be divided by visits. THIS IS NOT MOVES LEFT! It represents game length!
    pub fn M(&self) -> f32 {
        self.M
    }

    pub fn add_M(&mut self, value: f32) {
        self.M += value
    }

    /// W is the sum of values of each child node. Needs to be divided by visits.
    pub fn W(&self) -> f32 {
        self.W
    }

    pub fn add_W(&mut self, value: f32) {
        self.W += value
    }

    pub fn is_unexpanded(&self) -> bool {
        self.node.is_unexpanded()
    }

    pub fn mark_as_expanding(&mut self) {
        self.node.mark_as_expanding()
    }

    pub fn set_expanded(&mut self, index: Index) {
        self.node.set_expanded(index);
    }

    pub fn get_waiter(&mut self) -> Rc<LocalManualResetEvent> {
        self.node.get_waiter()
    }

    pub fn clear(&mut self) {
        self.visits = 0;
        self.W = 0.0;
        self.M = 0.0;
    }
}

impl<A> From<ActionWithPolicy<A>> for MCTSEdge<A> {
    fn from(action_with_policy: ActionWithPolicy<A>) -> Self {
        MCTSEdge::new(action_with_policy.action, action_with_policy.policy_score)
    }
}

#[derive(Debug)]
enum MCTSNodeState {
    Unexpanded,
    Expanding,
    ExpandingWithWaiters(Rc<LocalManualResetEvent>),
    Expanded(Index),
}

impl MCTSNodeState {
    fn get_index(&self) -> Option<Index> {
        if let Self::Expanded(index) = self {
            Some(*index)
        } else {
            None
        }
    }

    fn is_unexpanded(&self) -> bool {
        matches!(self, Self::Unexpanded)
    }

    fn mark_as_expanding(&mut self) {
        debug_assert!(matches!(self, Self::Unexpanded));
        *self = Self::Expanding
    }

    fn set_expanded(&mut self, index: Index) {
        debug_assert!(!matches!(self, Self::Unexpanded));
        debug_assert!(!matches!(self, Self::Expanded(_)));
        let state = std::mem::replace(self, Self::Expanded(index));
        if let Self::ExpandingWithWaiters(reset_events) = state {
            reset_events.set()
        }
    }

    fn get_waiter(&mut self) -> Rc<LocalManualResetEvent> {
        match self {
            Self::Expanding => {
                let reset_event = Rc::new(LocalManualResetEvent::new(false));
                *self = Self::ExpandingWithWaiters(reset_event.clone());
                reset_event
            }
            Self::ExpandingWithWaiters(reset_event) => reset_event.clone(),
            _ => panic!("Node state is not currently expanding"),
        }
    }
}
