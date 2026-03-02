use std::cmp::max;
use std::thread;

use crossbeam::channel;

use super::node_graph_store::NodeGraphStore;
use crate::analysis_coordinator::{AnalysisCoordinator, InFlightExpansions};
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
    ) -> Self {
        let virtual_sims = virtual_sims.max(1);
        let store: PuctStore<E, VM> = NodeGraphStore::new();
        let context_pool = SearchContextPool::new(virtual_sims);

        Self {
            game_engine,
            analyzer,
            value_model,
            selection_strategy,
            store,
            context_pool,
            virtual_sims,
        }
    }

    pub fn prune(&mut self, game_state: &E::State) {
        let transposition_hash = game_state.transposition_hash();
        self.store.prune_to_transposition_hash(transposition_hash);
    }

    pub fn search<Alive>(&mut self, game_state: &E::State, alive: Alive) -> usize
    where
        Alive: FnMut(NodeInfo) -> bool + Send,
    {
        let root = self.get_or_create_root(game_state);
        self.run_simulations(root, game_state, alive)
    }

    fn run_simulations<Alive>(&self, root: NodeId, game_state: &E::State, alive: Alive) -> usize
    where
        Alive: FnMut(NodeInfo) -> bool + Send,
    {
        let mut max_depth = 0;
        thread::scope(|s| {
            let (tx, rx) = channel::bounded::<SimMsgOf<E, VM>>(self.virtual_sims);

            let (expansions, coordinator) =
                AnalysisCoordinator::new(self.analyzer, self.virtual_sims);

            let analyzer = self.analyzer;
            let store = &self.store;
            let value_model = self.value_model;
            let bp_expansions = expansions.clone();

            let backprop_handle = s.spawn(move || {
                let backprop = Backpropagator::new(analyzer, store, value_model, bp_expansions, rx);
                backprop.run();
            });

            let sim_handle = s.spawn(|| self.run_sim(root, game_state, alive, tx, expansions));

            let coordinator_handle = s.spawn(move || coordinator.run());

            max_depth = sim_handle.join().expect("PUCT sim thread panicked");
            backprop_handle.join().expect("backprop thread panicked");
            coordinator_handle
                .join()
                .expect("in-flight coordinator panicked");
        });

        max_depth
    }

    fn run_sim<Alive>(
        &self,
        root: NodeId,
        game_state: &E::State,
        mut alive: Alive,
        tx: SimTx<E, VM>,
        exp: impl InFlightExpansions<State = E::State>,
    ) -> usize
    where
        Alive: FnMut(NodeInfo) -> bool,
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
            let root_info = NodeInfo {
                visits: root_node.visits(),
                virtual_visits: root_node.virtual_visits(),
                depth: max_depth as u32,
            };

            if !alive(root_info) {
                break;
            }

            let step = simulator.simulate_once(root, game_state.clone(), sim_id);
            max_depth = max(max_depth, step.depth());

            self.handle_step(step, &tx, &exp);

            sim_id += 1;
        }

        max_depth
    }

    fn handle_step(
        &self,
        step: SimulationStep<E::State, E::Terminal>,
        tx: &SimTx<E, VM>,
        exp: &impl InFlightExpansions<State = E::State>,
    ) {
        let msg = match step {
            SimulationStep::Terminal(step) => {
                let terminal_snapshot = self.value_model.terminal_snapshot(&step.terminal);

                SimMsg::new_terminal(step.sim_id, step.path, terminal_snapshot)
            }
            SimulationStep::NewLeaf(step) => {
                let hash = step.game_state.transposition_hash();
                exp.analyze(hash, step.game_state.clone());

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

        let request_id = transposition_hash;
        self.analyzer.analyze(request_id, game_state);
        let (recv_request_id, analysis) = self.analyzer.recv();

        assert!(
            recv_request_id == request_id,
            "Expected analysis result for root node"
        );

        let (policy_priors, predictions) = analysis.into_inner();
        let snapshot = self.value_model.pred_snapshot(&predictions);
        let rollup = snapshot.into();

        store.create_and_insert_state_node(transposition_hash, policy_priors, rollup)
    }
}

// @TODO: Solve cycles
// @TODO: Add repetition count to hash
