use dashmap::DashMap;
use engine::GameEngine;
use half::f16;
use model::Analyzer;
use smallvec::SmallVec;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::Ordering;
use std::thread::JoinHandle;

use super::{BackpropagationStrategy, EdgeInfo, NodeArena, NodeId, SelectionPolicy};

const NUM_SELECTIONS: i32 = 10;
const BATCH_SIZE: usize = 32;

pub struct PUCT<'a, E, M, B, Sel>
where
    M: Analyzer,
    B: BackpropagationStrategy,
    E: GameEngine,
{
    game_engine: &'a E,
    analyzer: &'a M,
    backpropagation_strategy: &'a B,
    selection_strategy: &'a Sel,
    game_state: E::State,
    nodes: NodeArena<PUCTNode<E::Action, M::Predictions, B::RollupStats>>,
    transposition_table: DashMap<u64, NodeId>,
}

struct PUCTNode<A, P, R> {
    transposition_hash: u64,
    visits: u32,
    predictions: P,
    rollup_stats: R,
    edges: Vec<PUCTEdge<A>>,
}

struct PUCTEdge<A> {
    action: A,
    policy_prior: f16,
    visits: u32,
    /// Acts as a cached child node. Requires validation to support path-dependent and stochastic state transitions.
    child: AtomicU32,
}

impl<A> PUCTEdge<A> {
    fn as_edge_info<R>(&self) -> EdgeInfo<A, R> {
        EdgeInfo {
            action: &self.action,
            policy_prior: self.policy_prior.into(),
            visits: self.visits,
            rollup_stats: None,
        }
    }

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
        self.child.store(new_value, Ordering::Relaxed);
    }
}

impl<E, M, B, Sel> PUCT<'_, E, M, B, Sel>
where
    E: GameEngine,
    M: Analyzer,
    B: BackpropagationStrategy,
    Sel: SelectionPolicy<Action = E::Action, State = E::State, RollupStats = B::RollupStats>,
{
    pub fn search(&mut self) {
        self.run_simulate();
    }

    fn run_simulate(&self) {
        let root_node_id = NodeId::from_u32(0);
        let selection = self.select_leaf(root_node_id);
        let expansion = self.evaluate_leaf(&selection);
        self.apply_expansion(expansion);
    }

    /// Get child from edge if cached, otherwise lookup in transposition table and cache.
    /// Validates the cached child by comparing transposition hashes.
    /// Returns None if this is a new position that needs expansion.
    fn get_or_lookup_child(&self, edge: &PUCTEdge<E::Action>, state: &E::State) -> Option<NodeId> {
        let transposition_hash = self.game_engine.hash(state);

        // Check if cached child matches this hash
        if let Some(child_id) = edge.get_child() {
            let child_node = self.nodes.get(child_id);
            if child_node.transposition_hash == transposition_hash {
                return Some(child_id);
            }
        }

        // Cache miss or invalid, lookup in transposition table
        if let Some(existing_id) = self.transposition_table.get(&transposition_hash) {
            let existing_id = *existing_id;
            edge.set_child(existing_id);
            Some(existing_id)
        } else {
            None
        }
    }

    fn select_leaf(&self, node_id: NodeId) -> SelectionResult<E::State> {
        let mut path = vec![];
        let mut current = node_id;
        let mut state: E::State = self.game_state.clone();

        loop {
            let node = self.nodes.get(current);
            path.push(current);

            if node.edges.is_empty() {
                break;
            }

            let edge_idx = self.select_edge(node);
            let edge = &node.edges[edge_idx];

            state = self.game_engine.take_action(&state, &edge.action);

            if let Some(child_id) = self.get_or_lookup_child(edge, &state) {
                current = child_id;
            } else {
                break;
            }
        }

        SelectionResult {
            path,
            leaf_state: state,
        }
    }

    fn evaluate_leaf(
        &self,
        selection: &SelectionResult<E::State>,
    ) -> ExpansionResult<E::State, E::Action, M::Predictions, B::RollupStats> {
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

    fn apply_expansion(
        &mut self,
        result: ExpansionResult<E::State, E::Action, M::Predictions, B::RollupStats>,
    ) {
        let state_hash = self.game_engine.hash(&result.leaf_state);

        // Store hash in the new node
        let mut new_node = result.new_node;
        new_node.transposition_hash = state_hash;

        let new_node_id = self.nodes.push(new_node);
        self.transposition_table.insert(state_hash, new_node_id);

        if let Some(&parent_id) = result.path.last() {
            let parent = self.nodes.get(parent_id);
        }

        for &node_id in result.path.iter().rev() {
            let node = self.nodes.get(node_id);
            node.visits += 1;
        }
    }

    fn get_root_node(&self) -> &PUCTNode<E::Action, M::Predictions, B::RollupStats> {
        self.nodes.get(NodeId::from_u32(0))
    }

    fn select_edge(&self, node: &PUCTNode<E::Action, M::Predictions, B::RollupStats>) -> usize {
        self.selection_strategy
            .select_edge(&node.edges, node.visits, &self.game_state, 0)
    }
}

struct SelectionResult<S> {
    path: Vec<NodeId>,
    leaf_state: S,
}

struct ExpansionResult<S, A, P, R> {
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
// @TODO: Add repetition count to hash
// @TODO: Maybe from_u32 isn't always the root
