use std::sync::atomic::{AtomicU32, Ordering};

use super::NodeId;

pub struct PUCTEdge {
    pub visits: AtomicU32,
    pub child: AtomicU32,
}

impl PUCTEdge {
    pub fn new() -> Self {
        Self {
            visits: AtomicU32::new(0),
            child: AtomicU32::new(NodeId::unset().as_u32()),
        }
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
        self.child.store(new_value, Ordering::Relaxed);
    }

    pub fn try_set_child(&self, new_child: NodeId) -> bool {
        self.child
            .compare_exchange(
                NodeId::unset().as_u32(),
                new_child.as_u32(),
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
    }

    pub fn increment_visits(&self) {
        self.visits.fetch_add(1, Ordering::AcqRel);
    }
}

impl Default for PUCTEdge {
    fn default() -> Self {
        Self::new()
    }
}
