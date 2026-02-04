use common::TranspositionHash;
use dashmap::DashMap;
use engine::GameEngine;
use half::f16;
use model::ActionWithPolicy;
use model::GameAnalyzer;
use std::collections::HashSet;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::Ordering;
use std::thread::JoinHandle;

use super::{
    AfterState, BackpropagationStrategy, BorrowedOrOwned, EdgeInfo, NodeArena, NodeId, PUCTEdge,
    SelectionPolicy, StateNode, Terminal,
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
    nodes: NodeArena<StateNode<E::Action, B::RollupStats>, AfterState, Terminal<E::Terminal>>,
    transposition_table: DashMap<u64, NodeId>,
}

impl<E, M, B, Sel> PUCT<'_, E, M, B, Sel>
where
    E: GameEngine,
    E::Terminal: engine::Value,
    M: GameAnalyzer<State = E::State, Predictions = E::Terminal, Action = E::Action>,
    B: BackpropagationStrategy<State = E::State, Predictions = E::Terminal>,
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

    fn expand_and_backpropagate(
        &self,
        selection: SelectionResult<'_, E::State, E::Terminal, B::NodeInfo>,
    ) {
        let (path_info, predictions) = match selection {
            SelectionResult::Unexpanded(unexpanded) => self.expand_unexpanded(unexpanded),
            SelectionResult::Terminal(terminal) => terminal.into_inner(),
        };

        self.backpropagate(path_info, &predictions);
    }

    fn expand_unexpanded(
        &self,
        unexpanded: UnexpandedSelection<'_, E::State, B::NodeInfo>,
    ) -> (Vec<NodePathInfo<B::NodeInfo>>, M::Predictions) {
        let UnexpandedSelection {
            mut path_info,
            game_state,
        } = unexpanded;

        let (policy_priors, predictions) = self.analyzer.analyze(&*game_state).into_inner();
        let new_node_id = self.create_node(game_state.transposition_hash(), policy_priors);

        let node_info = self.backpropagation_strategy.node_info(&*game_state);
        path_info.push(NodePathInfo {
            node_id: new_node_id,
            node_info,
        });

        (path_info, predictions)
    }

    /// Get child from edge if cached, otherwise lookup in transposition table and cache.
    /// Returns None if this is a new position that needs expansion.
    fn get_or_lookup_child(&self, edge: &PUCTEdge, state: &E::State) -> Option<NodeId> {
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

    fn select_leaf(
        &self,
        node_id: NodeId,
    ) -> SelectionResult<'_, E::State, E::Terminal, B::NodeInfo> {
        let mut path_info = vec![];
        let mut visited = HashSet::new();
        let mut current = node_id;
        let mut game_state = BorrowedOrOwned::Borrowed(&self.game_state);

        loop {
            let node = self.nodes.get(current);

            if visited.insert(current) {
                let node_info = self.backpropagation_strategy.node_info(&*game_state);

                path_info.push(NodePathInfo {
                    node_id: current,
                    node_info,
                });

                node.visits.fetch_add(1, Ordering::AcqRel);
            }

            debug_assert!(
                !node.edges.is_empty(),
                "Node in tree should always have edges - should be terminal"
            );

            let edge_idx = self.select_edge(node);
            let edge = node
                .get_edge(edge_idx)
                .expect("Selected edge must be expanded");
            let action = &node.get_action(edge_idx).action;

            edge.visits.fetch_add(1, Ordering::AcqRel);

            let next_game_state = self.game_engine.take_action(&game_state, action);
            game_state = BorrowedOrOwned::Owned(next_game_state);

            if let Some(terminal_value) = self.game_engine.terminal_state(&game_state) {
                return SelectionResult::Terminal(TerminalSelection {
                    path_info,
                    terminal_value,
                });
            } else if let Some(child_id) = self.get_or_lookup_child(edge, &game_state) {
                current = child_id;
            } else {
                return SelectionResult::Unexpanded(UnexpandedSelection {
                    path_info,
                    game_state,
                });
            }
        }
    }

    fn create_node(
        &self,
        transposition_hash: u64,
        policy_priors: Vec<ActionWithPolicy<M::Action>>,
    ) -> NodeId {
        let new_node = StateNode::new(transposition_hash, policy_priors);

        let new_node_id = self.nodes.push(new_node);
        self.transposition_table
            .insert(transposition_hash, new_node_id);

        new_node_id
    }

    fn backpropagate(
        &self,
        path_info: Vec<NodePathInfo<B::NodeInfo>>,
        predictions: &M::Predictions,
    ) {
        for node_path in path_info.iter() {
            let node = self.nodes.get(node_path.node_id);
            self.backpropagation_strategy.backpropagate(
                &node_path.node_info,
                &node.rollup_stats,
                predictions,
            );
        }
    }

    fn get_root_node(&self) -> &StateNode<E::Action, B::RollupStats> {
        self.nodes.get(NodeId::from_u32(0))
    }

    fn select_edge(&self, node: &StateNode<E::Action, B::RollupStats>) -> usize {
        let edge_iter = (0..node.edge_count()).map(|i| {
            let edge = node.get_edge(i).unwrap();
            let action_with_policy = node.get_action(i);

            EdgeInfo {
                action: &action_with_policy.action,
                policy_prior: action_with_policy.policy,
                visits: edge.visits.load(Ordering::Acquire),
                rollup_stats: None,
            }
        });

        self.selection_strategy.select_edge(
            edge_iter,
            node.visits.load(Ordering::Acquire),
            &self.game_state,
            0,
        )
    }
}

struct NodePathInfo<NI> {
    node_id: NodeId,
    node_info: NI,
}

enum SelectionResult<'a, S, T, NI> {
    Terminal(TerminalSelection<T, NI>),
    Unexpanded(UnexpandedSelection<'a, S, NI>),
}

struct TerminalSelection<T, NI> {
    path_info: Vec<NodePathInfo<NI>>,
    terminal_value: T,
}

impl<T, NI> TerminalSelection<T, NI> {
    fn into_inner(self) -> (Vec<NodePathInfo<NI>>, T) {
        (self.path_info, self.terminal_value)
    }
}

struct UnexpandedSelection<'a, S, NI> {
    path_info: Vec<NodePathInfo<NI>>,
    game_state: BorrowedOrOwned<'a, S>,
}

struct ExpansionResult<S, A, P, R> {
    path: Vec<NodeId>,
    game_state: S,
    new_node: StateNode<A, R>,
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
