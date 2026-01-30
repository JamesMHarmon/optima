use crossbeam::channel::Receiver;
use crossbeam::channel::bounded;
use dashmap::DashMap;
use half::f16;
use model::Analyzer;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicBool};
use std::sync::atomic::Ordering;
use std::thread::JoinHandle;
use std::thread::spawn;

use super::{NodeArena, NodeId};

const NUM_SELECTIONS: i32 = 10;
const BATCH_SIZE: usize = 32;

pub struct PUCT<'a, S, A, M, E, B, Sel, P, R>
where
    M: Analyzer,
    B: BackpropagationStrategy,
{
    game_engine: &'a E,
    analyzer: &'a M,
    backpropagation_strategy: &'a B,
    selection_strategy: &'a Sel,
    game_state: S,
    nodes: NodeArena<PUCTNode<A, M::Predictions, B::RollupStats>>,
    transposition_table: DashMap<u64, NodeId>,
}

struct PUCTNode<A, P, R> {
    visits: u32,
    predictions: P,
    rollup_stats: R,
    edges: Vec<PUCTEdge<A>>,
}

struct PUCTEdge<A> {
    action: A,
    policy_prior: f16,
    visits: u32,
    child: AtomicU32
}

impl<A> PUCTEdge<A> {
    fn get_child(&self) -> Option<NodeId> {
        let raw = self.child.load(Ordering::Acquire);
        if raw == u32::MAX {
            None
        } else {
            Some(NodeId::from_u32(raw))
        }
    }
    
    fn set_child(&self, node_id: NodeId) {
        let new_value = node_id.as_u32();
        
        debug_assert!({
            let old = self.child.load(Ordering::Relaxed);
            old == u32::MAX || old == new_value
        }, "Child mismatch");
        
        self.child.store(new_value, Ordering::Relaxed); 
    }
}

impl<S, A, M, E, B, Sel, P, R> PUCT<'_, S, A, M, E, B, Sel, P, R> {
    pub fn search(&mut self) {
        self.run_simulate();
    }

    fn run_simulate(&self) {
        let root_node_id = NodeId::from_usize(0);
        let selection = self.select_leaf(root_node_id);
        let expansion = self.evaluate_leaf(&selection);
        self.apply_expansion(expansion);
    }

    /// Get child from edge if cached, otherwise lookup in transposition table and cache.
    /// Returns None if this is a truly new position that needs expansion.
    fn get_or_lookup_child(&self, edge: &PUCTEdge<A>, state: &S) -> Option<NodeId> {
        if let Some(child_id) = edge.get_child() {
            return Some(child_id);
        }
        
        let child_hash = self.game_engine.hash(state);
        
        if let Some(existing_id) = self.transposition_table.get(&child_hash) {
            let existing_id = *existing_id;
            edge.set_child(existing_id);
            Some(existing_id)
        } else {
            None
        }
    }

    fn select_leaf(&self, node_id: NodeId) -> SelectionResult<S> {
        let mut path = vec![];
        let mut current = node_id;
        let mut state: S = self.game_state.clone();
        
        loop {
            let node = self.nodes.get(current);
            path.push(current);
            
            if node.edges.is_empty() {
                break;
            }
            
            let edge_idx = self.select_best_edge(node);
            let edge = &node.edges[edge_idx];
            
            state = self.game_engine.take_action(&state, &edge.action);
            
            if let Some(child_id) = self.get_or_lookup_child(edge, &state) {
                current = child_id;
            } else {
                break;
            }
        }
        
        SelectionResult { path, leaf_state: state }
    }

    fn evaluate_leaf(&self, selection: &SelectionResult<S>) -> ExpansionResult<A, P, R> {
        let analysis = self.analyzer.get_state_analysis(&selection.leaf_state);
        let new_node = self.create_node_from_analysis(analysis);
        let value = self.evaluate_node(&new_node);
        
        ExpansionResult {
            path: selection.path.clone(),
            leaf_state: selection.leaf_state.clone(),
            new_node,
            value,
        }
    }

    fn apply_expansion(&mut self, result: ExpansionResult<A, P, R>) {
        let new_node_id = self.nodes.push(result.new_node);
        
        let state_hash = self.game_engine.hash(&result.leaf_state);
        self.transposition_table.insert(state_hash, new_node_id);
        
        if let Some(&parent_id) = result.path.last() {
            let parent = self.nodes.get_mut(parent_id);
        }
        
        for &node_id in result.path.iter().rev() {
            let node = self.nodes.get_mut(node_id);
            node.visits += 1;
        }
    }

    fn get_root_node(&self) -> &PUCTNode<A, P, R> {
        self.nodes.get(NodeId::from_usize(0))
    }

    fn select() {}
}


struct SelectionResult<S> {
    path: Vec<NodeId>,
    leaf_state: S,
}

struct ExpansionResult<A, P, R> {
    path: Vec<NodeId>,
    leaf_state: S,
    new_node: PUCTNode<A, P, R>,
    value: f32,
}

struct Workers {
    handles: Vec<JoinHandle<()>>,
}

impl Workers {
    fn join(self) {
        for h in self.handles {
            h.join().unwrap();
        }
    }
}

// Multi-thread implementationr
// Read: trace down the tree to find nodes to expand
// - checks if the node is in a cache?
// - put a node in the cache if not already
// - have n number of threads

// Write: deterministic trace that updates node/edge values and backpropagate results up the tree
// Write: expands nodes

//@ TOODO: Solve cycles
//@ TODO: Add proper child node average value updates
//@ TODO: When applying an expansion, need to link the parent edge to the new child node
//@ TODO: Check for and reduce clones
