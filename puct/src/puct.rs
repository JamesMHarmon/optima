use std::cmp::max;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::thread;

use crossbeam::channel;

use super::node_graph_store::NodeGraphStore;
use crate::analysis_coordinator::AnalysisCoordinator;
use crate::backprop::{Backpropagator, SimMsg};
use crate::node_arena::NodeId;
use crate::search_context::SearchContextPool;
use crate::selection_policy::SelectionPolicy;
use crate::selection_policy::{EdgeScore, SelectionPolicyScoring};
use crate::simulate::Simulator;
use crate::value_model::ValueModel;
use crate::{NodeInfo, SimulationStep};
use common::TranspositionHash;
use engine::GameEngine;
use model::GameAnalyzer;

type RollupOf<VM> = <VM as ValueModel>::Rollup;
type SnapshotOf<VM> = <VM as ValueModel>::Snapshot;
type PuctStore<E, VM> = NodeGraphStore<<E as GameEngine>::Action, RollupOf<VM>>;

type SimMsgOf<E, VM> = SimMsg<<E as GameEngine>::State, SnapshotOf<VM>>;
type SimTx<E, VM> = channel::Sender<SimMsgOf<E, VM>>;

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
    virtual_sims: usize,
    sim_threads: usize,
}

impl<'a, E, M, VM, Sel> PUCT<'a, E, M, VM, Sel>
where
    E: GameEngine + Sync,
    M: GameAnalyzer<State = E::State, Action = E::Action> + Sync,
    VM: ValueModel<Predictions = M::Predictions, Terminal = E::Terminal> + Sync,
    RollupOf<VM>: Send + Sync,
    Sel: SelectionPolicy<
            SnapshotOf<VM>,
            State = E::State,
            Action = E::Action,
            Terminal = E::Terminal,
        > + Sync,
    E::State: TranspositionHash + Clone + Send + Sync,
    E::Action: Send + Sync,
    SnapshotOf<VM>: Send + Sync,
{
    pub fn new(
        game_engine: &'a E,
        analyzer: &'a M,
        value_model: &'a VM,
        selection_strategy: &'a Sel,
        virtual_sims: usize,
        sim_threads: usize,
    ) -> Self {
        let virtual_sims = virtual_sims.max(1);
        let sim_threads = sim_threads.max(1);
        let store: PuctStore<E, VM> = NodeGraphStore::new();
        let context_pool = SearchContextPool::new(virtual_sims * sim_threads);

        Self {
            game_engine,
            analyzer,
            value_model,
            selection_strategy,
            store,
            context_pool,
            virtual_sims,
            sim_threads,
        }
    }

    pub fn prune(&mut self, game_state: &E::State) {
        let transposition_hash = game_state.transposition_hash();
        self.store.prune_to_transposition_hash(transposition_hash);
    }

    pub fn search<Alive>(&mut self, game_state: &E::State, alive: Alive) -> usize
    where
        Alive: Fn(NodeInfo) -> bool + Send + Sync,
    {
        let root = self.get_or_create_root(game_state);
        self.run_simulations(root, game_state, alive)
    }

    fn run_simulations<Alive>(&self, root: NodeId, game_state: &E::State, alive: Alive) -> usize
    where
        Alive: Fn(NodeInfo) -> bool + Send + Sync,
    {
        let sim_id = Arc::new(AtomicUsize::new(0));
        let total_capacity = self.virtual_sims * self.sim_threads;
        let coordinator = AnalysisCoordinator::new(self.analyzer, total_capacity);
        let (tx, rx) = channel::bounded::<SimMsgOf<E, VM>>(total_capacity);
        let mut max_depth = 0;

        thread::scope(|s| {
            let analyzer = self.analyzer;
            let store = &self.store;
            let value_model = self.value_model;
            let coordinator = &coordinator;
            let alive = &alive;

            let backprop_handle = s.spawn(move || {
                let backprop = Backpropagator::new(analyzer, store, value_model, coordinator, rx);
                backprop.run();
            });

            let mut sim_handles = Vec::with_capacity(self.sim_threads);
            for _ in 0..self.sim_threads {
                let tx = tx.clone();
                let sim_id = sim_id.clone();
                let handle =
                    s.spawn(move || self.run_sim(root, game_state, alive, tx, coordinator, sim_id));
                sim_handles.push(handle);
            }

            drop(tx);

            let depths = sim_handles.into_iter().map(|h| h.join());
            max_depth = depths
                .map(|d| d.expect("PUCT sim thread panicked"))
                .max()
                .unwrap_or(0);

            backprop_handle.join().expect("backprop thread panicked");
        });

        max_depth
    }

    fn run_sim<Alive>(
        &self,
        root: NodeId,
        game_state: &E::State,
        alive: &Alive,
        tx: SimTx<E, VM>,
        coordinator: &AnalysisCoordinator<M>,
        sim_id: Arc<AtomicUsize>,
    ) -> usize
    where
        Alive: Fn(NodeInfo) -> bool,
    {
        let root_node = self.store.state_node(root);
        let mut max_depth = 0;

        let mut simulator = Simulator::new(
            &self.store,
            self.game_engine,
            self.selection_strategy,
            &self.context_pool,
        );

        loop {
            let root_info = NodeInfo {
                visits: root_node.visits(),
                virtual_visits: root_node.virtual_visits(),
                depth: max_depth as u32,
            };

            if !alive(root_info) {
                break;
            }

            let current_sim_id = sim_id.fetch_add(1, AtomicOrdering::Relaxed);
            let step = simulator.simulate_once(root, game_state.clone(), current_sim_id);
            max_depth = max(max_depth, step.depth());

            self.handle_step(step, &tx, coordinator);
        }

        max_depth
    }

    fn handle_step(
        &self,
        step: SimulationStep<E::State, E::Terminal>,
        tx: &SimTx<E, VM>,
        coordinator: &AnalysisCoordinator<M>,
    ) {
        let msg = match step {
            SimulationStep::Terminal(step) => {
                let terminal_snapshot = self.value_model.terminal_snapshot(&step.terminal);

                SimMsg::new_terminal(step.sim_id, step.path, terminal_snapshot)
            }
            SimulationStep::NewLeaf(step) => {
                coordinator.analyze(step.game_state.clone());

                SimMsg::new_state(step.sim_id, step.game_state, step.path)
            }
        };

        let _ = tx.send(msg);
    }

    /// Returns an owned snapshot of edge stats suitable for UIs/wrappers.
    pub fn edge_views(&mut self, game_state: &E::State) -> Vec<EdgeView<E::Action, SnapshotOf<VM>>>
    where
        E::Action: Clone,
    {
        let transposition_hash = game_state.transposition_hash();

        if self.store.get_node_id(transposition_hash).is_none() {
            self.get_or_create_root(game_state);
        }

        self.try_edge_views(game_state)
            .expect("Node has been created")
    }

    pub fn try_edge_views(
        &mut self,
        game_state: &E::State,
    ) -> Option<Vec<EdgeView<E::Action, SnapshotOf<VM>>>>
    where
        E::Action: Clone,
    {
        let transposition_hash = game_state.transposition_hash();
        let node_id = self.store.get_node_id(transposition_hash)?;

        let node = self.store.state_node(node_id);
        node.ensure_frontier_edge();
        Some(
            self.store
                .iter_edge_info(node)
                .map(|e| EdgeView {
                    edge_index: e.edge_index,
                    action: e.action.clone(),
                    policy_prior: e.policy_prior,
                    visits: e.visits,
                    snapshot: e.snapshot,
                })
                .collect(),
        )
    }

    /// Computes per-edge PUCT scores for the given node, using the active selection policy.
    ///
    /// This is intended for UI/debug output (e.g. filling `EdgeDetails`). It uses the
    /// same scoring terms as selection, including virtual-loss handling.
    pub fn edge_scores(&self, game_state: &E::State, depth: u32) -> Vec<EdgeScore>
    where
        Sel: SelectionPolicyScoring<SnapshotOf<VM>, State = E::State>,
    {
        let transposition_hash = game_state.transposition_hash();
        let Some(node_id) = self.store.get_node_id(transposition_hash) else {
            return Vec::new();
        };

        let node = self.store.state_node(node_id);
        node.ensure_frontier_edge();

        self.selection_strategy.score_edges(
            NodeInfo {
                visits: node.visits(),
                virtual_visits: node.virtual_visits(),
                depth,
            },
            self.store.iter_edge_info(node),
            game_state,
        )
    }

    fn get_or_create_root(&mut self, game_state: &E::State) -> NodeId {
        let store = &self.store;
        let transposition_hash = game_state.transposition_hash();
        if let Some(existing) = store.get_node_id(transposition_hash) {
            return existing;
        }

        let analysis = self.analyzer.analyze(game_state);

        let (policy_priors, predictions) = analysis.into_inner();
        let snapshot = self.value_model.pred_snapshot(&predictions);
        let rollup = snapshot.into();

        store.create_and_insert_state_node(transposition_hash, policy_priors, rollup)
    }
}
