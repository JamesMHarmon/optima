use super::{
    AfterState, BackpropagationStrategy, BorrowedOrOwned, EdgeRef, NodeArena, NodeGraph, NodeId,
    PUCTEdge, RollupStats, SearchContextGuard, SearchContextPool, SelectionPolicy, StateNode,
    Terminal,
};
use common::TranspositionHash;
use dashmap::DashMap;
use engine::GameEngine;
use model::ActionWithPolicy;
use model::GameAnalyzer;

type PUCTNodeArena<A, R, SI> = NodeArena<StateNode<A, R, SI>, AfterState, Terminal<R>>;

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
    nodes: PUCTNodeArena<E::Action, B::RollupStats, B::StateInfo>,
    graph: NodeGraph<'a, E::Action, B::RollupStats, B::StateInfo>,
    transposition_table: DashMap<u64, NodeId>,
    context_pool: SearchContextPool,
}

impl<E, M, B, Sel> PUCT<'_, E, M, B, Sel>
where
    E: GameEngine,
    M: GameAnalyzer<State = E::State, Predictions = E::Terminal, Action = E::Action>,
    B: BackpropagationStrategy<State = E::State, Predictions = E::Terminal>,
    B::RollupStats: RollupStats,
    Sel: SelectionPolicy<B::RollupStats, State = E::State>,
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

    fn select_leaf(&self, node_id: NodeId) -> SelectionResult<'_, E::State, E::Terminal> {
        let mut ctx = self.context_pool.acquire();
        let (path, visited) = ctx.split_mut();
        let mut current = node_id;
        let mut game_state = BorrowedOrOwned::Borrowed(&self.game_state);

        loop {
            let node = self.nodes.get_state_node(current);

            if visited.insert(current) {
                path.push(current);
                node.increment_visits();
            }

            let edge_idx = self.select_edge(&game_state, node);
            let (edge, action) = node.edge_and_action(edge_idx);

            edge.increment_visits();

            let next_game_state = self.game_engine.take_action(&game_state, action);
            game_state = BorrowedOrOwned::Owned(next_game_state);

            if let Some(terminal) = self.game_engine.terminal_state(&game_state) {
                return SelectionResult::new(ctx, edge, game_state, Some(terminal));
            }

            if let Some(child_id) =
                self.get_or_link_transposition(edge, game_state.transposition_hash())
            {
                current = child_id;
                continue;
            }

            return SelectionResult::new(ctx, edge, game_state, None);
        }
    }

    fn backpropagate(&self, path: &[NodeId]) {
        for &node_id in path.iter().rev() {
            let node = self.nodes.get_state_node(node_id);
            // @TODO: Need to include a nodes own prediction in the rollup, otherwise we won't update nodes that are only visited once and never have a child added (e.g. leaf nodes that hit a cutoff)
            let aggregated = B::RollupStats::aggregate_weighted(
                node.iter_edge_snapshots(&self.nodes)
                    .filter_map(|(edge, snapshot)| snapshot.map(|s| (s, edge.visits()))),
            );

            node.rollup_stats().set(&aggregated);
        }
    }

    fn select_edge(
        &self,
        game_state: &E::State,
        node: &StateNode<E::Action, B::RollupStats, B::StateInfo>,
    ) -> usize {
        // @TODO: Set Depth
        node.ensure_frontier_edge();
        self.selection_strategy.select_edge(
            node.iter_edge_info(&self.nodes),
            node.visits(),
            game_state,
            0,
        )
    }

    fn expand_and_backpropagate(&self, selection: SelectionResult<'_, E::State, E::Terminal>) {
        let new_node = match &selection.terminal {
            None => Some(self.analyze_and_create_node(&selection.game_state)),
            Some(terminal) => {
                self.create_or_merge_terminal(selection.edge, &selection.game_state, terminal)
            }
        };

        if let Some(new_node_id) = new_node {
            self.graph.add_child_to_edge(selection.edge, new_node_id);
        }

        self.backpropagate(selection.path());
    }

    fn create_node(
        &self,
        transposition_hash: u64,
        policy_priors: Vec<ActionWithPolicy<M::Action>>,
        game_state: &E::State,
        predictions: &M::Predictions,
    ) -> NodeId {
        debug_assert!(
            !policy_priors.is_empty(),
            "Cannot create state node without actions - should be terminal"
        );

        let state_info = self.backpropagation_strategy.state_info(game_state);
        let rollup_stats = self.create_rollup_stats(game_state, predictions);
        let new_node = StateNode::new(transposition_hash, policy_priors, state_info, rollup_stats);

        let new_node_id = self.nodes.push_state(new_node);
        let previous_entry = self
            .transposition_table
            .insert(transposition_hash, new_node_id);

        debug_assert!(
            previous_entry.is_none(),
            "Transposition table entry for hash already exists"
        );

        new_node_id
    }

    fn create_rollup_stats(
        &self,
        game_state: &E::State,
        predictions: &M::Predictions,
    ) -> B::RollupStats {
        let state_info = self.backpropagation_strategy.state_info(game_state);
        self.backpropagation_strategy
            .create_rollup_stats(&state_info, predictions)
    }

    fn analyze_and_create_node(&self, game_state: &E::State) -> NodeId {
        let (policy_priors, predictions) = self.analyzer.analyze(game_state).into_inner();

        self.create_node(
            game_state.transposition_hash(),
            policy_priors,
            game_state,
            &predictions,
        )
    }

    fn update_edge_with_terminal(
        &self,
        edge: &PUCTEdge,
        rollup_stats: B::RollupStats,
    ) -> Option<NodeId> {
        if let Some((terminal_id, visits)) = self.graph.find_edge_terminal(edge) {
            let terminal_rollup = self.nodes.get_terminal_node(terminal_id).rollup_stats();
            let terminal_visits = visits - 1; // Subtract 1 since we already incremented visits for this edge during selection
            terminal_rollup.merge_rollup_weighted(terminal_visits, &rollup_stats, 1);
            None
        } else {
            let terminal_id = self.nodes.push_terminal(Terminal::new(rollup_stats));
            Some(terminal_id)
        }
    }

    /// Get child from edge if cached, otherwise lookup in transposition table and link.
    /// Returns None if this is a new position that needs expansion.
    fn get_or_link_transposition(
        &self,
        edge: &PUCTEdge,
        transposition_hash: u64,
    ) -> Option<NodeId> {
        if let Some(nested_child_id) = self
            .graph
            .get_edge_state_with_hash(edge, transposition_hash)
        {
            return Some(nested_child_id);
        }

        if let Some(existing_id) = self.transposition_table.get(&transposition_hash) {
            let existing_id = *existing_id;
            self.graph.add_child_to_edge(edge, existing_id);
            Some(existing_id)
        } else {
            None
        }
    }

    fn create_or_merge_terminal(
        &self,
        edge: &PUCTEdge,
        game_state: &E::State,
        terminal: &E::Terminal,
    ) -> Option<NodeId> {
        let rollup_stats = self.create_rollup_stats(game_state, terminal);
        self.update_edge_with_terminal(edge, rollup_stats)
    }
}

struct SelectionResult<'a, S, T> {
    context: SearchContextGuard,
    edge: EdgeRef<'a>,
    game_state: BorrowedOrOwned<'a, S>,
    terminal: Option<T>,
}

impl<'a, S, T> SelectionResult<'a, S, T> {
    fn new(
        context: SearchContextGuard,
        edge: EdgeRef<'a>,
        game_state: BorrowedOrOwned<'a, S>,
        terminal: Option<T>,
    ) -> Self {
        Self {
            context,
            edge,
            game_state,
            terminal,
        }
    }

    fn path(&self) -> &[NodeId] {
        &self.context.get_ref().path
    }
}

// Multi-thread implementation
// Read: trace down the tree to find nodes to expand
// - checks if the node is in a cache?
// - put a node in the cache if not already
// - have n number of threads

// Write: deterministic trace that updates node/edge values and backpropagate results up the tree
// Write: expands nodes

// @TODO: Solve cycles
// @TODO: Add proper child node average value updates
// @TODO: When applying an expansion, need to link the parent edge to the new child node
// @TODO: Check for and reduce clones
// @TODO: Add repetition count to hash
// @TODO: Maybe from_u32 isn't always the root
