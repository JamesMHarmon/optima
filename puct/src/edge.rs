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
            child: AtomicU32::new(u32::MAX),
        }
    }

    pub fn get_child(&self) -> Option<NodeId> {
        let raw = self.child.load(Ordering::Acquire);
        if raw == u32::MAX {
            None
        } else {
            Some(NodeId::from_u32(raw))
        }
    }

    pub fn set_child(&self, node_id: NodeId) {
        let new_value = node_id.as_u32();
        self.child.store(new_value, Ordering::Relaxed);
    }
}
