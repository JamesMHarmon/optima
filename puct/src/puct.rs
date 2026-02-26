use std::cmp::max;
use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;

use crossbeam::channel;

use super::node_graph_store::NodeGraphStore;
use crate::backprop::{BackpropTask, Backpropagator, SimMsg};
use crate::edge::PUCTEdge;
use crate::node_arena::NodeId;
use crate::rollup::RollupStats;
use crate::search_context::SearchContextPool;
use crate::selection_policy::SelectionPolicy;
use crate::simulate::{NewLeafStep, SimulationStep, Simulator, TerminalStep, WaiterInfo};
use crate::value_model::ValueModel;
use common::TranspositionHash;
use engine::GameEngine;
use model::{ActionWithPolicy, GameAnalyzer};

type RollupOf<VM> = <VM as ValueModel>::Rollup;
type SnapshotOf<VM> = <RollupOf<VM> as RollupStats>::Snapshot;
type PuctStore<E, VM> = NodeGraphStore<<E as GameEngine>::Action, RollupOf<VM>>;

type SimTx<S> = channel::Sender<SimMsg<S>>;

/// Capacity of the sim→backprop channel. Controls how far ahead the simulation
/// thread can run before it blocks waiting for the backprop thread to catch up.
const ANALYSIS_PIPELINE_DEPTH: usize = 16;

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
    M: GameAnalyzer<State = E::State, Action = E::Action, RequestId = usize> + Sync,
    VM: ValueModel<State = E::State, Predictions = M::Predictions, Terminal = E::Terminal> + Sync,
    RollupOf<VM>: Send + Sync,
    Sel: SelectionPolicy<SnapshotOf<VM>, State = E::State> + Sync,
    E::State: TranspositionHash + Clone + Send + Sync,
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
        let root = self.get_or_create_root(game_state);
        self.run_simulations(root, game_state, alive, &alive_flag)
    }

    fn run_simulations<Alive>(
        &self,
        root: NodeId,
        game_state: &E::State,
        alive: Alive,
        alive_flag: &AtomicBool,
    ) -> usize
    where
        Alive: FnMut(usize) -> bool + Send,
    {
        let (tx, rx) = channel::bounded::<SimMsg<E::State>>(ANALYSIS_PIPELINE_DEPTH * 4);

        let mut max_depth = 0;
        thread::scope(|s| {
            let backprop = Backpropagator::new(self.analyzer, &self.store, self.value_model);
            let backprop_handle = s.spawn(move || backprop.run(rx));
            let sim_handle = s.spawn(|| self.run_sim(root, game_state, alive, alive_flag, tx));

            max_depth = sim_handle.join().expect("PUCT sim thread panicked");
            backprop_handle.join().expect("backprop thread panicked");
        });

        max_depth
    }

    fn run_sim<Alive>(
        &self,
        root: NodeId,
        game_state: &E::State,
        mut alive: Alive,
        alive_flag: &AtomicBool,
        tx: SimTx<E::State>,
    ) -> usize
    where
        Alive: FnMut(usize) -> bool,
    {
        let root_node = self.store.state_node(root);
        let mut max_depth = 0;
        let mut sim_id: usize = 0;

        let mut simulator = Simulator::new(
            &self.store,
            self.game_engine,
            self.selection_strategy,
            &self.context_pool,
        );

        loop {
            let node_visits = root_node.visits() as usize;

            if !alive(node_visits) {
                alive_flag.store(false, Ordering::SeqCst);
                break;
            }

            let step = simulator.simulate_once(root, game_state.clone(), sim_id);
            max_depth = max(max_depth, step.depth());

            self.handle_step(step, &tx);

            sim_id += 1;
        }

        max_depth
    }

    fn handle_step(&self, step: SimulationStep<E::State, E::Terminal>, tx: &SimTx<E::State>) {
        match step {
            SimulationStep::Terminal(terminal_step) => self.handle_terminal(terminal_step, tx),
            SimulationStep::NewLeaf(new_leaf_step) => self.handle_new_leaf(new_leaf_step, tx),
        }
    }

    fn handle_terminal(&self, step: TerminalStep<E::State, E::Terminal>, tx: &SimTx<E::State>) {
        let parent = self.store.state_node(step.parent_node_id);
        let (edge, _) = parent.edge_and_action(step.edge_index);
        let new_node_id = self.create_or_merge_terminal(edge, &step.game_state, &step.terminal);
        let _ = tx.send(SimMsg::Terminal {
            sim_id: step.sim_id,
            task: BackpropTask {
                path: step.path,
                parent_node_id: step.parent_node_id,
                edge_index: step.edge_index,
                new_node_id,
            },
        });
    }

    fn handle_new_leaf(&self, step: NewLeafStep<E::State>, tx: &SimTx<E::State>) {
        // @TODO fix this
        let in_flight_hashes = &mut HashSet::new();
        let waiter = WaiterInfo {
            sim_id: step.sim_id,
            path: step.path,
            parent_node_id: step.parent_node_id,
            edge_index: step.edge_index,
        };
        if in_flight_hashes.contains(&step.transposition_hash) {
            let _ = tx.send(SimMsg::Waiter {
                hash: step.transposition_hash,
                waiter,
            });
        } else {
            in_flight_hashes.insert(step.transposition_hash);
            self.analyzer.analyze(step.sim_id, &step.game_state);
            let _ = tx.send(SimMsg::NewLeaf {
                request_id: step.sim_id,
                hash: step.transposition_hash,
                game_state: step.game_state,
                waiter,
            });
        }
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

    fn get_or_create_root(&mut self, game_state: &E::State) -> NodeId {
        let transposition_hash = game_state.transposition_hash();
        if let Some(existing) = self.store.get_node_id(transposition_hash) {
            return existing;
        }

        let request_id = transposition_hash as usize;
        self.analyzer.analyze(request_id, game_state);
        let (recv_request_id, analysis) = self.analyzer.recv();

        assert!(
            recv_request_id == request_id,
            "Expected analysis result for root node"
        );

        let (policy_priors, predictions) = analysis.into_inner();
        self.create_state_node(transposition_hash, policy_priors, game_state, &predictions)
    }

    fn create_state_node(
        &self,
        transposition_hash: u64,
        policy_priors: Vec<ActionWithPolicy<M::Action>>,
        game_state: &E::State,
        predictions: &M::Predictions,
    ) -> NodeId {
        let snapshot = self.value_model.pred_snapshot(game_state, predictions);
        let rollup_stats = snapshot.into();
        self.store
            .create_and_insert_state_node(transposition_hash, policy_priors, rollup_stats)
    }

    fn update_edge_with_terminal(
        &self,
        edge: &PUCTEdge,
        snapshot: SnapshotOf<VM>,
    ) -> Option<NodeId> {
        if let Some(terminal_id) = self.store.graph().find_edge_terminal(edge) {
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

// @TODO: Solve cycles
// @TODO: Add repetition count to hash
