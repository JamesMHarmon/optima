use std::sync::atomic::{AtomicU32, Ordering};

use crate::{EdgeInfo, node_arena::NodeId};

pub struct PUCTEdge {
    visits: AtomicU32,
    virtual_visits: AtomicU32,
    child: AtomicU32,
}

impl Default for PUCTEdge {
    fn default() -> Self {
        Self {
            visits: AtomicU32::new(0),
            virtual_visits: AtomicU32::new(0),
            child: AtomicU32::new(NodeId::unset().as_u32()),
        }
    }
}

impl PUCTEdge {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn visits(&self) -> u32 {
        self.visits.load(Ordering::Acquire)
    }

    pub fn child(&self) -> Option<NodeId> {
        let raw = self.child.load(Ordering::Acquire);
        let node_id = NodeId::from_u32(raw);
        if node_id.is_unset() {
            None
        } else {
            Some(node_id)
        }
    }

    pub fn set_child(&self, node_id: NodeId) {
        let new_value = node_id.as_u32();
        self.child.store(new_value, Ordering::Release);
    }

    pub fn try_set_child(&self, new_child: NodeId) -> Result<(), NodeId> {
        self.child
            .compare_exchange(
                NodeId::unset().as_u32(),
                new_child.as_u32(),
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .map(|_| ())
            .map_err(NodeId::from_u32)
    }

    pub fn increment_visits(&self) {
        self.visits.fetch_add(1, Ordering::AcqRel);
    }

    pub fn increment_virtual_visits(&self) {
        self.virtual_visits.fetch_add(1, Ordering::AcqRel);
    }

    pub fn decrement_virtual_visits(&self) {
        let prev = self.virtual_visits.fetch_sub(1, Ordering::AcqRel);
        debug_assert!(prev > 0, "virtual_visits underflow");
    }

    pub fn virtual_visits(&self) -> u32 {
        self.virtual_visits.load(Ordering::Acquire)
    }
}

#[derive(Clone, Debug)]
pub struct EdgeView<A, SS> {
    pub edge_index: usize,
    pub action: A,
    pub policy_prior: f32,
    pub visits: u32,
    pub snapshot: Option<SS>,
}

impl<A, SS> From<EdgeInfo<'_, A, SS>> for EdgeView<A, SS>
where
    A: Clone,
{
    fn from(info: EdgeInfo<'_, A, SS>) -> Self {
        Self {
            edge_index: info.edge_index,
            action: info.action.clone(),
            policy_prior: info.policy_prior,
            visits: info.visits,
            snapshot: info.snapshot,
        }
    }
}
