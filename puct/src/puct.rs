use super::{
    BorrowedOrOwned, EdgeRef, NodeGraph, NodeId, PUCTEdge, RollupStats,
    SearchContextGuard, SearchContextPool, SelectionPolicy, StateNode, ValueModel,
};
use super::node_graph_store::NodeGraphStore;
use common::TranspositionHash;
use engine::GameEngine;
use model::ActionWithPolicy;
use model::GameAnalyzer;

type SnapshotOf<VM> = <<VM as ValueModel>::Rollup as RollupStats>::Snapshot;

type PuctStore<E, VM> = NodeGraphStore<<E as GameEngine>::Action, <VM as ValueModel>::Rollup>;

type PuctStateNode<E, VM> = StateNode<<E as GameEngine>::Action, <VM as ValueModel>::Rollup>;

pub struct PUCT<'a, E, M, VM, Sel>
where
    E: GameEngine,
    VM: ValueModel,
{
    game_engine: &'a E,
    analyzer: &'a M,
    value_model: &'a VM,
    selection_strategy: &'a Sel,
    store: PuctStore<E, VM>,
    context_pool: SearchContextPool,
}

impl<'a, E, M, VM, Sel> PUCT<'a, E, M, VM, Sel>
where
    E: GameEngine,
    M: GameAnalyzer<State = E::State, Action = E::Action>,
    VM: ValueModel<State = E::State, Predictions = M::Predictions, Terminal = E::Terminal>,
    Sel: SelectionPolicy<SnapshotOf<VM>, State = E::State>,
    E::State: TranspositionHash,
    E::Terminal: engine::Value,
{
    pub fn new(
        game_engine: &'a E,
        analyzer: &'a M,
        value_model: &'a VM,
        selection_strategy: &'a Sel,
    ) -> Self {
        let store: PuctStore<E, VM> = NodeGraphStore::new();
        let context_pool = SearchContextPool::new(32);

        Self {
            game_engine,
            analyzer,
            value_model,
            selection_strategy,
            store,
            context_pool,
        }
    }

    pub fn prune(&mut self, game_state: &E::State) {
        let transposition_hash = game_state.transposition_hash();
        self.store.prune_to_transposition_hash(transposition_hash);
    }

    pub fn search(&mut self, root: NodeId, game_state: &E::State) {
        self.run_simulate(root, game_state);
    }

    fn run_simulate(&self, root: NodeId, game_state: &E::State) {
        let selection = self.select_leaf(root, game_state);
        self.expand_and_backpropagate(selection);
    }

    #[inline]
    fn graph(&self) -> NodeGraph<'_, E::Action, <VM as ValueModel>::Rollup> {
        self.store.graph()
    }

    fn select_leaf<'s>(
        &self,
        node_id: NodeId,
        game_state: &'s E::State,
    ) -> SelectionResult<'_, 's, E::State, E::Terminal> {
        let store = &self.store;
        let game_engine = &self.game_engine;
        let mut ctx = self.context_pool.acquire();
        let (path, visited) = ctx.split_mut();
        let mut current = node_id;
        let mut game_state = BorrowedOrOwned::Borrowed(game_state);
        let mut depth = 0;

        loop {
            let node = store.state_node(current);

            if visited.insert(current) {
                path.push(current);
            }

            let edge_idx = self.select_edge(&game_state, node, depth);
            let (edge, action) = node.edge_and_action(edge_idx);

            let next_game_state = game_engine.take_action(&game_state, action);
            let terminal_state = game_engine.terminal_state(&next_game_state);
            let transposition_hash = next_game_state.transposition_hash();
            let is_terminal = terminal_state.is_some();

            game_state = BorrowedOrOwned::Owned(next_game_state);

            depth += 1;
            self.increment_selection_visits(node, edge, transposition_hash, is_terminal);

            if is_terminal {
                return SelectionResult::new(ctx, edge, game_state, terminal_state);
            }

            if let Some(child_id) = store.get_or_link_transposition(edge, transposition_hash) {
                current = child_id;
                continue;
            }

            return SelectionResult::new(ctx, edge, game_state, None);
        }
    }

    /// Increment selection-time visits for the chosen `(node, edge)`.
    ///
    /// - Always increments `node` and `edge` visits.
    /// - If `is_terminal`, increments the AfterState terminal outcome visits (if present).
    /// - Otherwise, increments the AfterState state outcome whose transposition hash matches
    ///   `transposition_hash` (if present).
    fn increment_selection_visits(
        &self,
        node: &PuctStateNode<E, VM>,
        edge: &PUCTEdge,
        transposition_hash: u64,
        is_terminal: bool,
    ) {
        let graph = self.graph();
        node.increment_visits();
        edge.increment_visits();

        if is_terminal {
            graph.increment_afterstate_terminal_visits(edge);
        } else {
            graph.increment_afterstate_visits(edge, transposition_hash);
        }
    }

    fn backpropagate(&self, path: &[NodeId]) {
        for &node_id in path.iter().rev() {
            self.store.recompute_rollup(node_id);
        }
    }

    fn select_edge(&self, game_state: &E::State, node: &PuctStateNode<E, VM>, depth: u32) -> usize {
        node.ensure_frontier_edge();
        self.selection_strategy.select_edge(
            self.store.iter_edge_info(node),
            node.visits(),
            game_state,
            depth,
        )
    }

    fn expand_and_backpropagate<'e, 's>(
        &self,
        selection: SelectionResult<'e, 's, E::State, E::Terminal>,
    ) {
        let state = &selection.game_state;
        let edge = selection.edge;

        let new_node = match &selection.terminal {
            None => Some(self.analyze_and_create_node(state)),
            Some(terminal) => self.create_or_merge_terminal(edge, state, terminal),
        };

        if let Some(new_node_id) = new_node {
            self.graph().add_child_to_edge(edge, new_node_id);
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
        let store = &self.store;
        let snapshot = self.value_model.pred_snapshot(game_state, predictions);
        let rollup_stats = snapshot.into();

        store.create_and_insert_state_node(transposition_hash, policy_priors, rollup_stats)
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
        snapshot: SnapshotOf<VM>,
    ) -> Option<NodeId> {
        if let Some(terminal_id) = self.graph().find_edge_terminal(edge) {
            let terminal_node = self.store.terminal_node(terminal_id);
            terminal_node.rollup_stats().accumulate(&snapshot);

            None
        } else {
            let rollup_stats = snapshot.into();
            let terminal_id = self.store.create_and_insert_terminal_node(rollup_stats);
            Some(terminal_id)
        }
    }

    fn create_or_merge_terminal(
        &self,
        edge: &PUCTEdge,
        state: &E::State,
        terminal: &E::Terminal,
    ) -> Option<NodeId> {
        let snapshot = self.value_model.terminal_snapshot(state, terminal);
        self.update_edge_with_terminal(edge, snapshot)
    }
}

struct SelectionResult<'e, 's, S, T> {
    context: SearchContextGuard,
    edge: EdgeRef<'e>,
    game_state: BorrowedOrOwned<'s, S>,
    terminal: Option<T>,
}

impl<'e, 's, S, T> SelectionResult<'e, 's, S, T> {
    fn new(
        context: SearchContextGuard,
        edge: EdgeRef<'e>,
        game_state: BorrowedOrOwned<'s, S>,
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
