use std::rc::Rc;

use futures_intrusive::sync::LocalManualResetEvent;
use generational_arena::Index;
use model::ActionWithPolicy;

use crate::edge::MCTSEdge;

#[derive(Debug)]
pub struct MCTSNode<A, V> {
    visits: usize,
    pub(crate) value_score: V,
    pub(crate) moves_left_score: f32,
    edges: Vec<MCTSEdge<A>>,
}

impl<A, V> MCTSNode<A, V> {
    pub fn new(
        value_score: V,
        policy_scores: Vec<ActionWithPolicy<A>>,
        moves_left_score: f32,
    ) -> Self {
        Self {
            visits: 1,
            value_score,
            moves_left_score,
            edges: policy_scores
                .into_iter()
                .map(|action_with_policy| MCTSEdge {
                    visits: 0,
                    W: 0.0,
                    M: 0.0,
                    action: action_with_policy.action,
                    policy_score: action_with_policy.policy_score,
                    node: MCTSNodeState::Unexpanded,
                })
                .collect(),
        }
    }

    pub fn get_node_visits(&self) -> usize {
        self.visits
    }

    pub fn get_child_by_index_mut(&mut self, index: usize) -> &mut MCTSEdge<A> {
        &mut self.edges[index]
    }

    pub fn is_terminal(&self) -> bool {
        self.edges.is_empty()
    }

    pub fn iter_edges(&self) -> impl Iterator<Item = &MCTSEdge<A>> {
        self.edges.iter()
    }

    pub fn child_len(&self) -> usize {
        self.edges.len()
    }

    pub fn iter_edges_mut(&mut self) -> impl Iterator<Item = &mut MCTSEdge<A>> {
        self.edges.iter_mut()
    }

    pub fn increment_visits(&mut self) {
        self.visits += 1;
    }

    pub fn visits(&self) -> usize {
        self.visits
    }

    pub fn set_visits(&mut self, visits: usize) {
        self.visits = visits;
    }

    pub fn get_parts_mut(&mut self) -> (&mut usize, &mut Vec<MCTSEdge<A>>) {
        (&mut self.visits, &mut self.edges)
    }
}

impl<A, V> MCTSNode<A, V>
where
    A: Eq,
{
    pub fn get_child_of_action(&self, action: &A) -> Option<&MCTSEdge<A>> {
        self.iter_edges().find(|c| c.action() == action)
    }

    pub fn get_position_of_action(&self, action: &A) -> Option<usize> {
        self.iter_edges().position(|c| c.action() == action)
    }
}

#[derive(Debug)]
pub enum MCTSNodeState {
    Unexpanded,
    Expanding,
    ExpandingWithWaiters(Rc<LocalManualResetEvent>),
    Expanded(Index),
}

impl MCTSNodeState {
    pub fn get_index(&self) -> Option<Index> {
        if let Self::Expanded(index) = self {
            Some(*index)
        } else {
            None
        }
    }

    pub fn is_unexpanded(&self) -> bool {
        matches!(self, Self::Unexpanded)
    }

    pub fn mark_as_expanding(&mut self) {
        debug_assert!(matches!(self, Self::Unexpanded));
        *self = Self::Expanding
    }

    pub fn set_expanded(&mut self, index: Index) {
        debug_assert!(!matches!(self, Self::Unexpanded));
        debug_assert!(!matches!(self, Self::Expanded(_)));
        let state = std::mem::replace(self, Self::Expanded(index));
        if let Self::ExpandingWithWaiters(reset_events) = state {
            reset_events.set()
        }
    }

    pub fn get_waiter(&mut self) -> Rc<LocalManualResetEvent> {
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
