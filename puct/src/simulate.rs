use common::TranspositionHash;
use engine::GameEngine;

use crate::NodeInfo;
use crate::node::StateNode;
use crate::node_arena::NodeId;
use crate::node_graph_store::NodeGraphStore;
use crate::rollup::RollupStats;
use crate::selection_policy::SelectionPolicy;
use crate::{PathStep, SearchContextPool, SelectionResult, SimulationStep};

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
    Sel: SelectionPolicy<R::Snapshot, State = E::State, Action = E::Action, Terminal = E::Terminal>,
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

        match result {
            SelectionResult::Terminal(term_res) => SimulationStep::new_terminal(
                sim_id,
                term_res.path().to_vec(),
                term_res.terminal,
                term_res.depth,
            ),
            SelectionResult::Leaf(leaf_res) => SimulationStep::new_leaf(
                sim_id,
                leaf_res.path().to_vec(),
                leaf_res.game_state,
                leaf_res.depth,
            ),
        }
    }

    fn select_leaf(
        &self,
        node_id: NodeId,
        game_state: E::State,
    ) -> SelectionResult<E::State, E::Terminal> {
        let store = self.store;
        let game_engine = self.game_engine;
        let sel_strat = self.selection_strategy;
        let mut ctx = self.context_pool.acquire();
        let (path, visited) = ctx.split_mut();

        let mut game_state = game_state;
        let mut current = node_id;
        let mut depth = 0;

        loop {
            let node = store.state_node(current);

            let edge_idx = self.select_edge(&game_state, node, depth as u32);
            let (edge, action) = node.edge_and_action(edge_idx);

            visited.insert(game_state.transposition_hash());
            path.push(PathStep {
                node_id: current,
                edge_index: edge_idx,
            });

            let traj_term = sel_strat.terminal_for_trajectory(&game_state, action, visited);

            game_state = game_engine.take_action(&game_state, action);
            let term_state = game_engine.terminal_state(&game_state);
            let term_state = term_state.or(traj_term);
            let transposition_hash = game_state.transposition_hash();

            depth += 1;

            node.increment_virtual_visits();
            edge.increment_virtual_visits();

            if let Some(term_state) = term_state {
                return SelectionResult::new_terminal(ctx, term_state, depth);
            }

            if let Some(child_id) = store.get_or_link_transposition_safe(edge, transposition_hash) {
                current = child_id;
                continue;
            }

            return SelectionResult::new_leaf(ctx, game_state, depth);
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
