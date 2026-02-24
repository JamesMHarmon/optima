use std::cmp::max;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;

use crossbeam::channel::{Receiver, Sender};

use super::node_graph_store::NodeGraphStore;
use super::probe::Probe;
use crate::borrowed_or_owned::BorrowedOrOwned;
use crate::edge::PUCTEdge;
use crate::node::{EdgeRef, StateNode};
use crate::node_arena::NodeId;
use crate::node_graph::NodeGraph;
use crate::rollup::RollupStats;
use crate::search_context::{SearchContextGuard, SearchContextPool};
use crate::selection_policy::SelectionPolicy;
use crate::value_model::ValueModel;
use common::TranspositionHash;
use engine::GameEngine;
use model::ActionWithPolicy;
use model::GameAnalyzer;

type RollupOf<VM> = <VM as ValueModel>::Rollup;

type SnapshotOf<VM> = <RollupOf<VM> as RollupStats>::Snapshot;

type PuctStore<E, VM> = NodeGraphStore<<E as GameEngine>::Action, RollupOf<VM>>;

type PuctStateNode<E, VM> = StateNode<<E as GameEngine>::Action, RollupOf<VM>>;

#[derive(Clone, Debug)]
pub struct EdgeView<A, SS> {
    pub edge_index: usize,
    pub action: A,
    pub policy_prior: f32,
    pub visits: u32,
    pub snapshot: Option<SS>,
}

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
    E: GameEngine + Sync,
    M: GameAnalyzer<State = E::State, Action = E::Action> + Sync,
    VM: ValueModel<State = E::State, Predictions = M::Predictions, Terminal = E::Terminal> + Sync,
    RollupOf<VM>: Send + Sync,
    Sel: SelectionPolicy<SnapshotOf<VM>, State = E::State> + Sync,
    E::State: TranspositionHash + Sync,
    E::Action: Send + Sync,
    SnapshotOf<VM>: Send + Sync,
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

    pub fn search<Alive>(&mut self, game_state: &E::State, alive: Alive) -> usize
    where
        Alive: FnMut(usize) -> bool + Send,
    {
        let alive_flag = AtomicBool::new(true);
        let (tx, rx) = crossbeam::channel::unbounded::<u64>();
        let root = self.get_or_create_root(game_state);
        let mut depth = 0;

        thread::scope(|s| {
            let handle = s.spawn(|| self.run_simulations(root, game_state, alive, &alive_flag, tx));
            self.run_probes(&alive_flag, rx);

            depth = handle.join().expect("PUCT search thread panicked");
        });

        depth
    }

    fn run_simulations<Alive>(
        &self,
        root: NodeId,
        game_state: &E::State,
        mut alive: Alive,
        alive_flag: &AtomicBool,
        search_leaf_tx: Sender<u64>,
    ) -> usize
    where
        Alive: FnMut(usize) -> bool + Send,
    {
        let root_node = self.store.state_node(root);
        let mut max_depth = 0;

        loop {
            let node_visits = root_node.visits() as usize;
            if !alive(node_visits) {
                alive_flag.store(false, Ordering::SeqCst);
                break;
            }

            let depth = self.simulate_once(root, game_state, &search_leaf_tx);
            max_depth = max(depth, max_depth);
        }

        max_depth
    }

    fn run_probes(&self, alive: &AtomicBool, search_leaf_rx: Receiver<u64>) {
        let probe = Probe::new(
            self.game_engine,
            self.analyzer,
            self.value_model,
            self.selection_strategy,
            search_leaf_rx,
        );

        probe.run(alive);
    }

    /// Returns an owned snapshot of edge stats suitable for UIs/wrappers.
    pub fn edge_views(&self, game_state: &E::State) -> Vec<EdgeView<E::Action, SnapshotOf<VM>>>
    where
        E::Action: Clone,
    {
        let transposition_hash = game_state.transposition_hash();
        let Some(node_id) = self.store.get_node_id(transposition_hash) else {
            return Vec::new();
        };

        let node = self.store.state_node(node_id);
        self.store
            .iter_edge_info(node)
            .map(|e| EdgeView {
                edge_index: e.edge_index,
                action: e.action.clone(),
                policy_prior: e.policy_prior,
                visits: e.visits,
                snapshot: e.snapshot,
            })
            .collect()
    }

    fn get_or_create_root(&self, game_state: &E::State) -> NodeId {
        let transposition_hash = game_state.transposition_hash();
        if let Some(existing) = self.store.get_node_id(transposition_hash) {
            return existing;
        }

        self.analyze_and_create_node(game_state)
    }

    fn simulate_once(&self, root: NodeId, game_state: &E::State, tx: &Sender<u64>) -> usize {
        let selection = self.select_leaf(root, game_state);
        self.expand_and_backpropagate(&selection);
        tx.send(selection.game_state.transposition_hash()).ok();
        selection.depth
    }

    #[inline]
    fn graph(&self) -> NodeGraph<'_, E::Action, RollupOf<VM>> {
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

            let edge_idx = self.select_edge(&game_state, node, depth as u32);
            let (edge, action) = node.edge_and_action(edge_idx);

            let next_game_state = game_engine.take_action(&game_state, action);
            let terminal_state = game_engine.terminal_state(&next_game_state);
            let transposition_hash = next_game_state.transposition_hash();
            let is_terminal = terminal_state.is_some();

            game_state = BorrowedOrOwned::Owned(next_game_state);

            depth += 1;
            self.increment_selection_visits(node, edge, transposition_hash, is_terminal);

            if is_terminal {
                return SelectionResult::new(ctx, edge, game_state, terminal_state, depth);
            }

            if let Some(child_id) = store.get_or_link_transposition(edge, transposition_hash) {
                current = child_id;
                continue;
            }

            return SelectionResult::new(ctx, edge, game_state, None, depth);
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
        selection: &SelectionResult<'e, 's, E::State, E::Terminal>,
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
    depth: usize,
}

impl<'e, 's, S, T> SelectionResult<'e, 's, S, T> {
    fn new(
        context: SearchContextGuard,
        edge: EdgeRef<'e>,
        game_state: BorrowedOrOwned<'s, S>,
        terminal: Option<T>,
        depth: usize,
    ) -> Self {
        Self {
            context,
            edge,
            game_state,
            terminal,
            depth,
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
// @TODO: Check for and reduce clones
// @TODO: Add repetition count to hash
