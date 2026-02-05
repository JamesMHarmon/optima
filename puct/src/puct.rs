use common::TranspositionHash;
use dashmap::DashMap;
use engine::GameEngine;
use model::ActionWithPolicy;
use model::GameAnalyzer;
use std::collections::HashSet;
use std::sync::atomic::{AtomicU32, Ordering};
use std::thread::JoinHandle;
use tinyvec::TinyVec;

use crate::node_arena;

use super::{
    AfterState, AfterStateOutcome, BackpropagationStrategy, BorrowedOrOwned, EdgeInfo, NodeArena, NodeId,
    PUCTEdge, SelectionPolicy, StateNode, Terminal, NodeType
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
    nodes: NodeArena<StateNode<E::Action, B::RollupStats, B::StateInfo>, AfterState, Terminal<B::RollupStats>>,
    transposition_table: DashMap<u64, NodeId>,
}

impl<E, M, B, Sel> PUCT<'_, E, M, B, Sel>
where
    E: GameEngine,
    M: GameAnalyzer<State = E::State, Predictions = E::Terminal, Action = E::Action>,
    B: BackpropagationStrategy<State = E::State, Predictions = E::Terminal>,
    Sel: SelectionPolicy<State = E::State>,
    E::State: TranspositionHash,
    E::Terminal: engine::Value,
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
        let (path, predictions) = match selection {
            SelectionResult::Unexpanded(unexpanded) => self.expand_unexpanded(unexpanded),
            SelectionResult::Terminal(terminal) => terminal.into_inner(),
        };

        self.backpropagate(path, &predictions);
    }

    fn expand_unexpanded(&self, unexpanded: UnexpandedSelection<'_, E::State>) -> (Vec<NodeId>, M::Predictions) {
        let UnexpandedSelection {
            mut path,
            game_state,
        } = unexpanded;

        let (policy_priors, predictions) = self.analyzer.analyze(&*game_state).into_inner();
        let new_node_id = self.create_node(game_state.transposition_hash(), policy_priors, &*game_state, &predictions);
        
        path.push(new_node_id);

        (path, predictions)
    }

    /// Get child from edge if cached, otherwise lookup in transposition table and cache.
    /// Returns None if this is a new position that needs expansion.
    fn get_or_set_transposition(&self, edge: &PUCTEdge, transposition_hash: u64) -> Option<NodeId> {
        if let Some(child_id) = edge.get_child() &&
            let Some(child_id) = self.find_referenced_state_node(child_id, transposition_hash) {
            return Some(child_id);
        }

        if let Some(existing_id) = self.transposition_table.get(&transposition_hash) {
            let existing_id = *existing_id;
            self.add_child_to_edge(edge, existing_id);
            Some(existing_id)
        } else {
            None
        }
    }

    fn find_referenced_state_node(&self, node_id: NodeId, transposition_hash: u64) -> Option<NodeId> {
        match node_id.node_type() {
            NodeType::State => (self.nodes.get_state(node_id).transposition_hash == transposition_hash).then_some(node_id),
            NodeType::AfterState => self.nodes.get_after_state(node_id).outcomes.iter().find_map(|outcome| {
                let child_id = NodeId::from_u32(outcome.child.load(Ordering::Acquire));
                self.find_referenced_state_node(child_id, transposition_hash)
            }),
            NodeType::Terminal => None
        }

        // @TODO: Do I need to increment visits if found in afterstate?
    }

    fn add_child_to_edge(&self, edge: &PUCTEdge, child_id: NodeId) {
        if edge.try_set_child(child_id) {
            return;
        }

        let existing_child_id = edge.get_child().expect("Child must be set if try_set_child failed");

        let after_state = match existing_child_id.node_type() {
            NodeType::AfterState => self.nodes.get_after_state(existing_child_id),
            NodeType::State | NodeType::Terminal => {
                // Convert to AfterState with existing child as first outcome
                let after_state_id = self.nodes.push_after_state(AfterState {
                    outcomes: TinyVec::new(),
                });
                
                let after_state = self.nodes.get_after_state(after_state_id);
                after_state.outcomes.push(AfterStateOutcome {
                    visits: AtomicU32::new(0),
                    child: AtomicU32::new(existing_child_id.as_u32()),
                });
                
                edge.set_child(after_state_id);
                after_state
            }
        };

        // Add new child as an outcome
        after_state.outcomes.push(AfterStateOutcome {
            visits: AtomicU32::new(0),
            child: AtomicU32::new(child_id.as_u32()),
        });

        debug_assert!(
            {
                let outcome_count = after_state.outcomes.len();
                let ids: HashSet<_> = after_state.outcomes.iter()
                    .map(|o| NodeId::from_u32(o.child.load(Ordering::Acquire)))
                    .collect();
                ids.len() == outcome_count && ids.iter().filter(|id| id.node_type() == NodeType::Terminal).count() <= 1
            },
            "AfterState outcomes must not contain duplicate node IDs and at most one terminal"
        );
    }


    fn select_leaf(&self, node_id: NodeId) -> SelectionResult<'_, E::State, E::Terminal> {
        let mut path = vec![];
        let mut visited = HashSet::new();
        let mut current = node_id;
        let mut game_state = BorrowedOrOwned::Borrowed(&self.game_state);

        loop {
            let node = self.nodes.get_state(current);

            if visited.insert(current) {
                path.push(current);
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
                self.update_edge_with_terminal(edge, &terminal_value);

                return SelectionResult::Terminal(TerminalSelection {
                    path,
                    terminal_value,
                });
            } else if let Some(child_id) = self.get_or_lookup_child(edge, game_state.transposition_hash()) {
                current = child_id;
            } else {
                return SelectionResult::Unexpanded(UnexpandedSelection {
                    path,
                    game_state,
                });
            }
        }
    }

    fn update_edge_with_terminal(&self, edge: &PUCTEdge, terminal_value: &E::Terminal) {
        // Update edge statistics based on terminal value
        // Placeholder for actual implementation
    }

    fn create_node(&self, transposition_hash: u64, policy_priors: Vec<ActionWithPolicy<M::Action>>, game_state: &E::State, predictions: &M::Predictions) -> NodeId {
        let state_info = self.backpropagation_strategy.state_info(game_state);
        let rollup_stats = self.backpropagation_strategy.create_rollup_stats(&state_info, predictions);
        let new_node = StateNode::new(transposition_hash, policy_priors, state_info, rollup_stats);

        let new_node_id = self.nodes.push_state(new_node);
        self.transposition_table
            .insert(transposition_hash, new_node_id);

        new_node_id
    }

    fn backpropagate(&self, path: Vec<NodeId>, _predictions: &M::Predictions) {
        for &node_id in path.iter().rev() {
            let node = self.nodes.get_state(node_id);
            
            self.backpropagation_strategy.aggregate_stats(
                &node.rollup_stats,
                node.iter_children_stats(&self.nodes),
            );
        }
    }

    fn get_root_node(&self) -> &StateNode<E::Action, B::RollupStats, B::StateInfo> {
        self.nodes.get_state(NodeId::from_u32(0))
    }

    fn select_edge(&self, node: &StateNode<E::Action, B::RollupStats, B::StateInfo>) -> usize {
        let edge_iter = (0..node.edge_count()).map(|i| {
            let edge = node.get_edge(i).unwrap();
            let action_with_policy = node.get_action(i);

            EdgeInfo::<E::Action, B::RollupStats> {
                action: &action_with_policy.action,
                policy_prior: action_with_policy.policy_score.to_f32(),
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

enum SelectionResult<'a, S, T> {
    Terminal(TerminalSelection<T>),
    Unexpanded(UnexpandedSelection<'a, S>),
}

struct TerminalSelection<T> {
    path: Vec<NodeId>,
    terminal_value: T,
}

impl<T> TerminalSelection<T> {
    fn into_inner(self) -> (Vec<NodeId>, T) {
        (self.path, self.terminal_value)
    }
}

struct UnexpandedSelection<'a, S> {
    path: Vec<NodeId>,
    game_state: BorrowedOrOwned<'a, S>,
}

struct ExpansionResult<S, A, P, R, SI> {
    path: Vec<NodeId>,
    game_state: S,
    new_node: StateNode<A, R, SI>,
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
