use append_only_vec::AppendOnlyVec;

#[repr(transparent)]
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct NodeId(u32);

impl NodeId {
    #[inline]
    pub fn as_u32(self) -> u32 {
        self.0
    }

    #[inline]
    pub(crate) fn from_u32(i: u32) -> Self {
        Self(i)
    }
}

pub struct NodeArena<T> {
    nodes: AppendOnlyVec<T>,
}

impl<T> NodeArena<T> {
    pub fn push(&self, node: T) -> NodeId {
        let i = self.nodes.push(node);
        NodeId::from_u32(i as u32)
    }

    #[inline]
    pub fn get(&self, id: NodeId) -> &T {
        &self.nodes[id.as_u32() as usize]
    }
}
