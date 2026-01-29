use append_only_vec::AppendOnlyVec;

#[repr(transparent)]
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct NodeId(usize);

impl NodeId {
    #[inline]
    pub fn as_usize(self) -> usize { self.0 }

    #[inline]
    pub(crate) fn from_usize(i: usize) -> Self { Self(i) }
}


pub struct NodeArena<T> {
    nodes: AppendOnlyVec<T>,
}

impl<T> NodeArena<T> {
    pub fn push(&self, node: T) -> NodeId {
        let i = self.nodes.push(node);
        NodeId::from_usize(i)
    }

    #[inline]
    pub fn get(&self, id: NodeId) -> &T {
        &self.nodes[id.as_usize()]
    }
}
