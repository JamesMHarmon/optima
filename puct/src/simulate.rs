use common::TranspositionHash;
use engine::GameEngine;

use crate::NodeInfo;
use crate::node::StateNode;
use crate::node_arena::NodeId;
use crate::node_graph_store::NodeGraphStore;
use crate::rollup::RollupStats;
use crate::search_context::{PathStep, SearchContextGuard, SearchContextPool};
use crate::selection_policy::SelectionPolicy;

type PuctStore<E, R> = NodeGraphStore<<E as GameEngine>::Action, R>;
type PuctStateNode<E, R> = StateNode<<E as GameEngine>::Action, R>;

/// Handles the tree-traversal (selection) phase of PUCT
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

    /// Run one simulation from `root`.
    pub(super) fn simulate_once(
        &mut self,
        root: NodeId,
        root_state: E::State,
        sim_id: usize,
    ) -> SimulationStep<E::State, E::Terminal> {
        let result = self.select_leaf(root, root_state);

        let depth = result.depth;
        let path = result.path().to_vec();
        let parent_node_id = result.parent_node_id;
        let edge_index = result.edge_index;
        let game_state = result.game_state;

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

        SimulationStep::NewLeaf(NewLeafStep {
            sim_id,
            path,
            parent_node_id,
            edge_index,
            game_state,
            depth,
        })
    }

    fn select_leaf(
        &self,
        node_id: NodeId,
        game_state: E::State,
    ) -> SelectionResult<E::State, E::Terminal> {
        let store = self.store;
        let game_engine = self.game_engine;
        let mut ctx = self.context_pool.acquire();
        let (path, _visited) = ctx.split_mut();

        let mut game_state = game_state;
        let mut current = node_id;
        let mut depth = 0;

        loop {
            let node = store.state_node(current);

            let edge_idx = self.select_edge(&game_state, node, depth as u32);
            let (edge, action) = node.edge_and_action(edge_idx);

            path.push(PathStep {
                node_id: current,
                edge_index: edge_idx,
            });

            game_state = game_engine.take_action(&game_state, action);
            let term_state = game_engine.terminal_state(&game_state);
            let transposition_hash = game_state.transposition_hash();
            let is_terminal = term_state.is_some();

            depth += 1;

            node.increment_virtual_visits();
            edge.increment_virtual_visits();

            if is_terminal {
                return SelectionResult::new(ctx, current, edge_idx, game_state, term_state, depth);
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
            NodeInfo {
                visits: node.visits(),
                virtual_visits: node.virtual_visits(),
                depth,
            },
            self.store.iter_edge_info(node),
            game_state,
        )
    }
}

pub(super) struct SelectionResult<S, T> {
    context: SearchContextGuard,
    pub(super) parent_node_id: NodeId,
    pub(super) edge_index: usize,
    pub(super) game_state: S,
    pub(super) terminal: Option<T>,
    pub(super) depth: usize,
}

impl<S, T> SelectionResult<S, T> {
    fn new(
        context: SearchContextGuard,
        parent_node_id: NodeId,
        edge_index: usize,
        game_state: S,
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

    pub(super) fn path(&self) -> &[PathStep] {
        &self.context.get_ref().path
    }
}

/// Describes the outcome of one simulation step.
pub(super) enum SimulationStep<S, T> {
    /// The leaf was a terminal state
    Terminal(TerminalStep<S, T>),
    /// A previously-unseen position was reached.
    NewLeaf(NewLeafStep<S>),
}

pub(super) struct TerminalStep<S, T> {
    pub(super) sim_id: usize,
    pub(super) path: Vec<PathStep>,
    pub(super) parent_node_id: NodeId,
    pub(super) edge_index: usize,
    pub(super) game_state: S,
    pub(super) terminal: T,
    pub(super) depth: usize,
}

pub(super) struct NewLeafStep<S> {
    pub(super) sim_id: usize,
    pub(super) path: Vec<PathStep>,
    pub(super) parent_node_id: NodeId,
    pub(super) edge_index: usize,
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
