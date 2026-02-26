use common::TranspositionHash;
use engine::GameEngine;

use crate::borrowed_or_owned::BorrowedOrOwned;
use crate::edge::PUCTEdge;
use crate::node::StateNode;
use crate::node_arena::NodeId;
use crate::node_graph_store::NodeGraphStore;
use crate::rollup::RollupStats;
use crate::search_context::{SearchContextGuard, SearchContextPool};
use crate::selection_policy::SelectionPolicy;

type PuctStore<E, R> = NodeGraphStore<<E as GameEngine>::Action, R>;
type PuctStateNode<E, R> = StateNode<<E as GameEngine>::Action, R>;

/// Handles the tree-traversal (selection) phase of PUCT and tracks which
/// positions are currently being expanded by outstanding network requests.
pub(super) struct Simulator<'a, E, R, Sel>
where
    E: GameEngine,
    R: RollupStats,
{
    store: &'a PuctStore<E, R>,
    game_engine: &'a E,
    selection_strategy: &'a Sel,
    context_pool: &'a SearchContextPool,
}

impl<'a, E, R, Sel> Simulator<'a, E, R, Sel>
where
    E: GameEngine,
    R: RollupStats,
    Sel: SelectionPolicy<R::Snapshot, State = E::State>,
    E::State: TranspositionHash,
{
    pub(super) fn new(
        store: &'a PuctStore<E, R>,
        game_engine: &'a E,
        selection_strategy: &'a Sel,
        context_pool: &'a SearchContextPool,
    ) -> Self {
        Self {
            store,
            game_engine,
            selection_strategy,
            context_pool,
        }
    }

    /// Run one simulation from `root`, returning a [`SimulationStep`] that
    /// describes what the caller must do next.
    ///
    /// For `Suspended` steps the expanding map has already been updated; no
    /// further action beyond depth tracking is needed.
    pub(super) fn simulate_once(
        &mut self,
        root: NodeId,
        root_state: &E::State,
        sim_id: usize,
    ) -> SimulationStep<E::State, E::Terminal>
    where
        E::State: Clone,
    {
        let result = self.select_leaf(root, root_state);

        let depth = result.depth;
        let path = result.path().to_vec();
        let parent_node_id = result.parent_node_id;
        let edge_index = result.edge_index;

        // @TODO: Why even borrow own here if always going to clone?
        // `game_state` is always Owned at the leaf because `take_action` is called
        // at least once before any return from `select_leaf`.
        let game_state = match result.game_state {
            BorrowedOrOwned::Owned(s) => s,
            BorrowedOrOwned::Borrowed(s) => s.clone(),
        };

        if let Some(terminal) = result.terminal {
            return SimulationStep::Terminal(TerminalStep {
                sim_id,
                path,
                parent_node_id,
                edge_index,
                game_state,
                terminal,
                depth,
            });
        }

        let transposition_hash = game_state.transposition_hash();

        // @TODO: Is there a race between suspend?
        SimulationStep::NewLeaf(NewLeafStep {
            sim_id,
            path,
            parent_node_id,
            edge_index,
            transposition_hash,
            game_state,
            depth,
        })
    }

    fn select_leaf<'s>(
        &self,
        node_id: NodeId,
        game_state: &'s E::State,
    ) -> SelectionResult<'s, E::State, E::Terminal> {
        let store = self.store;
        let game_engine = self.game_engine;
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
                return SelectionResult::new(
                    ctx,
                    current,
                    edge_idx,
                    game_state,
                    terminal_state,
                    depth,
                );
            }

            if let Some(child_id) = store.get_or_link_transposition(edge, transposition_hash) {
                current = child_id;
                continue;
            }

            return SelectionResult::new(ctx, current, edge_idx, game_state, None, depth);
        }
    }

    fn select_edge(&self, game_state: &E::State, node: &PuctStateNode<E, R>, depth: u32) -> usize {
        node.ensure_frontier_edge();
        self.selection_strategy.select_edge(
            self.store.iter_edge_info(node),
            node.visits(),
            game_state,
            depth,
        )
    }

    fn increment_selection_visits(
        &self,
        node: &PuctStateNode<E, R>,
        edge: &PUCTEdge,
        transposition_hash: u64,
        is_terminal: bool,
    ) {
        let graph = self.store.graph();
        node.increment_visits();
        edge.increment_visits();

        if is_terminal {
            graph.increment_afterstate_terminal_visits(edge);
        } else {
            graph.increment_afterstate_visits(edge, transposition_hash);
        }
    }
}

pub(super) struct SelectionResult<'s, S, T> {
    context: SearchContextGuard,
    pub(super) parent_node_id: NodeId,
    pub(super) edge_index: usize,
    pub(super) game_state: BorrowedOrOwned<'s, S>,
    pub(super) terminal: Option<T>,
    pub(super) depth: usize,
}

impl<'s, S, T> SelectionResult<'s, S, T> {
    fn new(
        context: SearchContextGuard,
        parent_node_id: NodeId,
        edge_index: usize,
        game_state: BorrowedOrOwned<'s, S>,
        terminal: Option<T>,
        depth: usize,
    ) -> Self {
        Self {
            context,
            parent_node_id,
            edge_index,
            game_state,
            terminal,
            depth,
        }
    }

    pub(super) fn path(&self) -> &[NodeId] {
        &self.context.get_ref().path
    }
}

/// Describes the outcome of one simulation step.
pub(super) enum SimulationStep<S, T> {
    /// The leaf was a terminal state; all needed data is immediately available.
    Terminal(TerminalStep<S, T>),
    /// A previously-unseen position was reached.
    ///
    /// The caller must call `analyzer.analyze(sim_id, &game_state)`.
    NewLeaf(NewLeafStep<S>),
}

pub(super) struct TerminalStep<S, T> {
    pub(super) sim_id: usize,
    pub(super) path: Vec<NodeId>,
    pub(super) parent_node_id: NodeId,
    pub(super) edge_index: usize,
    pub(super) game_state: S,
    pub(super) terminal: T,
    pub(super) depth: usize,
}

pub(super) struct NewLeafStep<S> {
    pub(super) sim_id: usize,
    pub(super) path: Vec<NodeId>,
    pub(super) parent_node_id: NodeId,
    pub(super) edge_index: usize,
    pub(super) transposition_hash: u64,
    pub(super) game_state: S,
    pub(super) depth: usize,
}

impl<S, T> SimulationStep<S, T> {
    pub(super) fn depth(&self) -> usize {
        match self {
            Self::Terminal(s) => s.depth,
            Self::NewLeaf(s) => s.depth,
        }
    }
}

/// Per-simulation data needed when the corresponding expansion result arrives.
pub(super) struct WaiterInfo {
    pub(super) sim_id: usize,
    pub(super) path: Vec<NodeId>,
    pub(super) parent_node_id: NodeId,
    pub(super) edge_index: usize,
}
