use common::TranspositionHash;
use common::transposition;
use dashmap::DashMap;
use engine::GameEngine;
use half::f16;
use model::GameAnalyzer;
use std::collections::HashSet;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::Ordering;
use std::thread::JoinHandle;

use super::{
    BackpropagationStrategy, BorrowedOrOwned, EdgeInfo, NodeArena, NodeId, SelectionPolicy,
};

const NUM_SELECTIONS: i32 = 10;
const BATCH_SIZE: usize = 32;

pub struct PUCT<'a, E, M, B, Sel>
where
    M: GameAnalyzer,
    B: BackpropagationStrategy,
    E: GameEngine,
{
    game_engine: &'a E,
    analyzer: &'a M,
    backpropagation_strategy: &'a B,
    selection_strategy: &'a Sel,
    game_state: E::State,
    nodes: NodeArena<PUCTNode<E::Action, B::RollupStats>>,
    transposition_table: DashMap<u64, NodeId>,
}

struct PUCTNode<A, R> {
    transposition_hash: u64,
    visits: u32,
    rollup_stats: R,
    edges: Vec<PUCTEdge<A>>,
}

struct PUCTEdge<A> {
    action: A,
    policy_prior: f16,
    visits: u32,
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
    E::Terminal: engine::Value,
    M: GameAnalyzer<State = E::State, Predictions = E::Terminal>,
    B: BackpropagationStrategy,
    Sel: SelectionPolicy<State = E::State>,
    E::State: TranspositionHash,
    B::RollupStats: Default,
{
    pub fn search(&mut self) {
        self.run_simulate();
    }

    fn run_simulate(&self) {
        let root_node_id = NodeId::from_u32(0);
        let selection = self.select_leaf(root_node_id);
        self.expand_and_backpropagate(selection);
    }

    fn expand_and_backpropagate(&self, selection: SelectionResult<'_, E::State, E::Terminal>) {
        match selection {
            SelectionResult::Unexpanded(unexpanded) => self.expand_unexpanded(unexpanded),
            SelectionResult::Terminal(terminal) => self.backpropagate_terminal(terminal),
        }
    }

    fn expand_unexpanded(&self, unexpanded: UnexpandedSelection<'_, E::State>) {
        let UnexpandedSelection {
            mut path,
            game_state,
        } = unexpanded;
        let analysis = self.analyzer.analyze(&*game_state);
        let new_node = self.create_node(&*game_state);
        path.push(new_node);

        self.backpropagate(path, analysis.predictions());
    }

    fn backpropagate_terminal(&self, terminal: TerminalSelection<'_, E::State, E::Terminal>) {
        let TerminalSelection {
            path,
            terminal_value,
            ..
        } = terminal;

        self.backpropagate(path, &terminal_value);
    }

    /// Get child from edge if cached, otherwise lookup in transposition table and cache.
    /// Returns None if this is a new position that needs expansion.
    fn get_or_lookup_child(&self, edge: &PUCTEdge<E::Action>, state: &E::State) -> Option<NodeId> {
        let transposition_hash = state.transposition_hash();

        if let Some(child_id) = edge.get_child() {
            let child_node = self.nodes.get(child_id);
            debug_assert_eq!(
                child_node.transposition_hash, transposition_hash,
                "Edge's cached child node must match the state hash - edges point to a single node"
            );
            return Some(child_id);
        }

        if let Some(existing_id) = self.transposition_table.get(&transposition_hash) {
            let existing_id = *existing_id;
            edge.set_child(existing_id);
            Some(existing_id)
        } else {
            None
        }
    }

    fn select_leaf(&self, node_id: NodeId) -> SelectionResult<'_, E::State, E::Terminal> {
        let mut path = vec![];
        let mut visited = HashSet::new();
        let mut current = node_id;
        let mut game_state = BorrowedOrOwned::Borrowed(&self.game_state);

        loop {
            let node = self.nodes.get(current);

            if visited.insert(current) {
                path.push(current);
            }

            debug_assert!(
                !node.edges.is_empty(),
                "Node in tree should always have edges - should be terminal"
            );

            let edge_idx = self.select_edge(node);
            let edge = &node.edges[edge_idx];

            let next_game_state = self.game_engine.take_action(&game_state, &edge.action);
            game_state = BorrowedOrOwned::Owned(next_game_state);

            if let Some(terminal_value) = self.game_engine.terminal_state(&game_state) {
                return SelectionResult::Terminal(TerminalSelection {
                    path,
                    game_state,
                    terminal_value,
                });
            } else if let Some(child_id) = self.get_or_lookup_child(edge, &game_state) {
                current = child_id;
            } else {
                return SelectionResult::Unexpanded(UnexpandedSelection { path, game_state });
            }
        }
    }

    fn create_node(&self, state: &E::State) -> NodeId {
        let transposition_hash = state.transposition_hash();

        let new_node = PUCTNode {
            transposition_hash,
            visits: 0,
            rollup_stats: B::RollupStats::default(),
            edges: vec![],
        };

        let new_node_id = self.nodes.push(new_node);
        self.transposition_table
            .insert(transposition_hash, new_node_id);

        return new_node_id;
    }

    fn backpropagate(&self, path: Vec<NodeId>, prediction: &M::Predictions) {
        // Backpropagate the known terminal value up the path
        // Path is already deduplicated by select_leaf
        for &node_id in path.iter().rev() {
            let node = self.nodes.get(node_id);
            node.visits += 1;
            // TODO: Update rollup stats with terminal value
        }
    }

    fn get_root_node(&self) -> &PUCTNode<E::Action, B::RollupStats> {
        self.nodes.get(NodeId::from_u32(0))
    }

    fn select_edge(&self, node: &PUCTNode<E::Action, B::RollupStats>) -> usize {
        let edge_iter = node
            .edges
            .iter()
            .map(|e| e.as_edge_info::<B::RollupStats>());
        self.selection_strategy
            .select_edge(edge_iter, node.visits, &self.game_state, 0)
    }
}

enum SelectionResult<'a, S, T> {
    Terminal(TerminalSelection<'a, S, T>),
    Unexpanded(UnexpandedSelection<'a, S>),
}

struct TerminalSelection<'a, S, T> {
    path: Vec<NodeId>,
    game_state: BorrowedOrOwned<'a, S>,
    terminal_value: T,
}

struct UnexpandedSelection<'a, S> {
    path: Vec<NodeId>,
    game_state: BorrowedOrOwned<'a, S>,
}

struct ExpansionResult<S, A, P, R> {
    path: Vec<NodeId>,
    game_state: S,
    new_node: PUCTNode<A, R>,
    predictions: P,
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
