use std::rc::Rc;

use futures_intrusive::sync::LocalManualResetEvent;
use generational_arena::Index;
use half::f16;
use model::ActionWithPolicy;

#[allow(non_snake_case)]
#[derive(Debug)]
pub struct MCTSEdge<A, PV> {
    action: A,
    visits: usize,
    policy_score: f16,
    propagatedValues: PV,
    node: MCTSNodeState,
}

#[allow(non_snake_case)]
impl<A, PV> MCTSEdge<A, PV>
{
    pub fn new(action: A, policy_score: f16) -> Self {
        Self {
            action,
            visits: 0,
            policy_score,
            propagatedValues,
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
        self.policy_score.to_f32()
    }

    pub fn set_policy_score(&mut self, policy_score: f32) {
        self.policy_score = f16::from_f32(policy_score)
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

    pub fn features(&self) -> &F {
        &self.features
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
}

impl<A, PV> MCTSEdge<A, PV>
where
    PV: Default,
{
    pub fn new(action: A, policy_score: f16) -> Self {
        Self {
            action,
            visits: 0,
            policy_score,
            propagatedValues,
            node: MCTSNodeState::Unexpanded,
        }
    }

    pub fn clear(&mut self) {
        self.visits = 0;
        self.propagatedValues = PV::default();
    }
}

impl<A, F> From<ActionWithPolicy<A>> for MCTSEdge<A, F>
where
    F: Default,
{
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
