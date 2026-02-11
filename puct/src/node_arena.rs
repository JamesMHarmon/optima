use append_only_vec::AppendOnlyVec;

#[repr(transparent)]
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct NodeId(u32);

// Node type encoding in top 2 bits:
// 11 = State node
// 01 = AfterState node
// 10 = Terminal node
// 00 = Unset/invalid
const TYPE_MASK: u32 = 0b11 << 30;
const INDEX_MASK: u32 = (1 << 30) - 1;
const STATE_TYPE: u32 = 0b11 << 30;
const AFTER_STATE_TYPE: u32 = 0b01 << 30;
const TERMINAL_TYPE: u32 = 0b10 << 30;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum NodeType {
    State,
    AfterState,
    Terminal,
}

impl NodeId {
    #[inline]
    pub fn as_u32(self) -> u32 {
        self.0
    }

    #[inline]
    pub fn node_type(self) -> NodeType {
        match self.0 & TYPE_MASK {
            STATE_TYPE => NodeType::State,
            AFTER_STATE_TYPE => NodeType::AfterState,
            TERMINAL_TYPE => NodeType::Terminal,
            _ => unreachable!("Invalid node type bits"),
        }
    }

    #[inline]
    pub fn is_state(self) -> bool {
        matches!(self.node_type(), NodeType::State)
    }

    #[inline]
    pub fn is_after_state(self) -> bool {
        matches!(self.node_type(), NodeType::AfterState)
    }

    #[inline]
    pub fn is_terminal(self) -> bool {
        matches!(self.node_type(), NodeType::Terminal)
    }

    #[inline]
    pub const fn unset() -> Self {
        Self(u32::MAX)
    }

    #[inline]
    pub const fn is_unset(self) -> bool {
        self.0 == u32::MAX
    }

    #[inline]
    fn index(self) -> usize {
        (self.0 & INDEX_MASK) as usize
    }

    #[inline]
    fn from_type_and_index(node_type: NodeType, index: usize) -> Self {
        debug_assert!(index < (1 << 30), "Index too large for NodeId");
        let type_bits = match node_type {
            NodeType::State => STATE_TYPE,
            NodeType::AfterState => AFTER_STATE_TYPE,
            NodeType::Terminal => TERMINAL_TYPE,
        };
        Self(type_bits | (index as u32))
    }

    #[inline]
    pub(crate) fn from_u32(i: u32) -> Self {
        Self(i)
    }
}

impl From<NodeId> for u32 {
    #[inline]
    fn from(id: NodeId) -> u32 {
        id.0
    }
}

impl From<u32> for NodeId {
    #[inline]
    fn from(value: u32) -> Self {
        Self::from_u32(value)
    }
}

impl From<NodeId> for usize {
    #[inline]
    fn from(id: NodeId) -> usize {
        id.index()
    }
}

pub struct NodeArena<S, A, T> {
    state_nodes: AppendOnlyVec<S>,
    after_state_nodes: AppendOnlyVec<A>,
    terminal_nodes: AppendOnlyVec<T>,
}

impl<S, A, T> NodeArena<S, A, T> {
    pub fn new() -> Self {
        Self {
            state_nodes: AppendOnlyVec::new(),
            after_state_nodes: AppendOnlyVec::new(),
            terminal_nodes: AppendOnlyVec::new(),
        }
    }

    pub fn push_state(&self, node: S) -> NodeId {
        let index = self.state_nodes.push(node);
        NodeId::from_type_and_index(NodeType::State, index)
    }

    pub fn push_after_state(&self, node: A) -> NodeId {
        let index = self.after_state_nodes.push(node);
        NodeId::from_type_and_index(NodeType::AfterState, index)
    }

    pub fn push_terminal(&self, node: T) -> NodeId {
        let index = self.terminal_nodes.push(node);
        NodeId::from_type_and_index(NodeType::Terminal, index)
    }

    #[inline]
    pub fn get_state_node(&self, id: NodeId) -> &S {
        debug_assert!(!id.is_unset(), "NodeId is unset");
        debug_assert_eq!(id.node_type(), NodeType::State);
        &self.state_nodes[id.index()]
    }

    #[inline]
    pub fn get_after_state_node(&self, id: NodeId) -> &A {
        debug_assert!(!id.is_unset(), "NodeId is unset");
        debug_assert_eq!(id.node_type(), NodeType::AfterState);
        &self.after_state_nodes[id.index()]
    }

    #[inline]
    pub fn get_after_state_node_mut(&mut self, id: NodeId) -> &mut A {
        debug_assert!(!id.is_unset(), "NodeId is unset");
        debug_assert_eq!(id.node_type(), NodeType::AfterState);
        &mut self.after_state_nodes[id.index()]
    }

    #[inline]
    pub fn get_terminal_node(&self, id: NodeId) -> &T {
        debug_assert!(!id.is_unset(), "NodeId is unset");
        debug_assert_eq!(id.node_type(), NodeType::Terminal);
        &self.terminal_nodes[id.index()]
    }

    pub fn into_vecs(self) -> (Vec<S>, Vec<A>, Vec<T>) {
        (
            self.state_nodes.into_vec(),
            self.after_state_nodes.into_vec(),
            self.terminal_nodes.into_vec(),
        )
    }
}

impl Default for NodeArena<(), (), ()> {
    fn default() -> Self {
        Self::new()
    }
}
