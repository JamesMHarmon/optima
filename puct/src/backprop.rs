use core::panic;
use std::collections::{BTreeMap, HashMap, HashSet};

use crossbeam::channel::Receiver;

use crate::analysis_coordinator::InFlightExpansions;
use crate::node_arena::NodeId;
use crate::node_graph_store::NodeGraphStore;
use crate::rollup::RollupStats;
use crate::search_context::PathStep;
use crate::value_model::ValueModel;
use common::TranspositionHash;
use model::{ActionWithPolicy, GameAnalyzer, GameStateAnalysis, RequestId};

type PuctStore<M, R> = NodeGraphStore<<M as GameAnalyzer>::Action, R>;
type RollupOf<VM> = <VM as ValueModel>::Rollup;
type SimRx<S> = Receiver<SimMsg<S>>;

/// Owns the receive-and-backprop loop; intended to run on a dedicated thread.
///
/// This implementation processes one simulation message at a time by blocking
/// on `rx.recv()`. Backpropagation is strictly ordered by `sim_id`: messages
/// can arrive out of order and are buffered, but we only process the next
/// expected `sim_id`.
///
/// When the next simulation is a [`SimMsg::State`], this thread blocks until
/// the matching network analysis arrives via [`GameAnalyzer::recv`]. Any
/// out-of-order analysis results are buffered in a `BTreeMap` keyed by request
/// id.
pub(super) struct Backpropagator<'a, M, VM, IE>
where
    M: GameAnalyzer,
    VM: ValueModel,
    IE: InFlightExpansions<State = M::State> + 'a,
{
    analyzer: &'a M,
    store: &'a PuctStore<M, RollupOf<VM>>,
    value_model: &'a VM,
    expansions: IE,
    rx: SimRx<M::State>,

    analysis_buffer: HashMap<RequestId, GameStateAnalysis<M::Action, M::Predictions>>,
    pending_by_sim_id: BTreeMap<usize, PendingSim<M::State>>,
    next_sim_id: usize,
    sim_done: bool,
}

impl<'a, M, VM, IE> Backpropagator<'a, M, VM, IE>
where
    M: GameAnalyzer,
    VM: ValueModel<State = M::State, Predictions = M::Predictions>,
    RollupOf<VM>: RollupStats,
    M::State: TranspositionHash + Clone,
    M::Action: Send + Sync,
    IE: InFlightExpansions<State = M::State> + 'a,
{
    pub(super) fn new(
        analyzer: &'a M,
        store: &'a PuctStore<M, RollupOf<VM>>,
        value_model: &'a VM,
        expansions: IE,
        rx: SimRx<M::State>,
    ) -> Self {
        Self {
            analyzer,
            store,
            value_model,
            expansions,
            rx,

            analysis_buffer: HashMap::new(),
            pending_by_sim_id: BTreeMap::new(),
            next_sim_id: 0,
            sim_done: false,
        }
    }

    pub(super) fn run(mut self) {
        while let Some(pending) = self.next_sim() {
            if let PendingSim::State(p) = &pending {
                let game_state = &p.game_state;
                let hash = p.game_state.transposition_hash();

                self.expansions.complete(hash);

                if self.store.get_node_id(hash).is_some() {
                    // Another simulation has already expanded this node and backpropagated the path, so we can skip it.
                    continue;
                }

                let analysis = self.recv_analysis_for(hash);

                let (policy_priors, predictions) = analysis.into_inner();
                let new_node_id =
                    self.create_state_node(hash, policy_priors, game_state, &predictions);
                self.link_child(p.parent_node_id, p.edge_index, new_node_id);
            }

            self.backprop_path(pending.path());
            self.remove_virtual_loss(pending.path());
        }
    }

    fn next_sim(&mut self) -> Option<PendingSim<M::State>> {
        loop {
            if let Some(sim) = self.try_next_sim() {
                return Some(sim);
            }

            let pending = &mut self.pending_by_sim_id;

            if self.sim_done {
                if pending.is_empty() {
                    return None;
                } else {
                    panic!(
                        "Simulation thread has finished but there are still pending simulations: {:?}",
                        pending.keys()
                    );
                }
            }

            match self.rx.recv() {
                Ok(msg) => {
                    let sim_id = msg.sim_id();
                    pending.insert(sim_id, PendingSim::from_msg(msg));
                }
                Err(_) => self.sim_done = true,
            }
        }
    }

    fn try_next_sim(&mut self) -> Option<PendingSim<M::State>> {
        let pending = &mut self.pending_by_sim_id;

        if let Some((&min_id, _)) = pending.first_key_value() {
            if min_id == self.next_sim_id {
                let (_, pending) = pending.pop_first()?;
                self.next_sim_id += 1;
                return Some(pending);
            }
        }

        None
    }

    fn recv_analysis_for(
        &mut self,
        request_id: RequestId,
    ) -> GameStateAnalysis<M::Action, M::Predictions> {
        loop {
            if let Some(analysis) = self.analysis_buffer.remove(&request_id) {
                return analysis;
            }

            let (recv_request_id, analysis) = self.analyzer.recv();
            if recv_request_id == request_id {
                return analysis;
            }

            self.analysis_buffer.insert(recv_request_id, analysis);
        }
    }

    fn link_child(&self, parent_node_id: NodeId, edge_index: usize, new_node_id: NodeId) {
        let store = self.store;

        let parent = store.state_node(parent_node_id);
        let (edge, _) = parent.edge_and_action(edge_index);
        store.graph().add_child_to_edge(edge, new_node_id);
    }

    fn backprop_path(&self, path: &[PathStep]) {
        let store = self.store;
        let mut seen = HashSet::with_capacity(path.len());
        for step in path.iter().rev() {
            if seen.insert(step.node_id) {
                store.recompute_rollup(step.node_id);
            }
        }
    }

    fn remove_virtual_loss(&self, path: &[PathStep]) {
        let store = self.store;
        for step in path {
            let node = store.state_node(step.node_id);
            node.decrement_virtual_visits();
            node.edge(step.edge_index).decrement_virtual_visits();
        }
    }

    pub(super) fn create_state_node(
        &self,
        transposition_hash: u64,
        policy_priors: Vec<ActionWithPolicy<M::Action>>,
        game_state: &M::State,
        predictions: &M::Predictions,
    ) -> NodeId {
        let snapshot = self.value_model.pred_snapshot(game_state, predictions);
        let rollup_stats = snapshot.into();
        self.store
            .create_and_insert_state_node(transposition_hash, policy_priors, rollup_stats)
    }
}

enum PendingSim<S> {
    Terminal(PendingTerminal),
    State(PendingState<S>),
}

impl<S> PendingSim<S> {
    fn from_msg(msg: SimMsg<S>) -> Self {
        match msg {
            SimMsg::Terminal { path, .. } => Self::Terminal(PendingTerminal { path }),
            SimMsg::State {
                game_state,
                path,
                parent_node_id,
                edge_index,
                ..
            } => Self::State(PendingState {
                game_state,
                path,
                parent_node_id,
                edge_index,
            }),
        }
    }

    fn path(&self) -> &Vec<PathStep> {
        match self {
            Self::Terminal(p) => &p.path,
            Self::State(p) => &p.path,
        }
    }
}

struct PendingTerminal {
    path: Vec<PathStep>,
}

struct PendingState<S> {
    game_state: S,
    path: Vec<PathStep>,
    parent_node_id: NodeId,
    edge_index: usize,
}

/// Message sent from the simulation thread to the backprop thread.
pub(super) enum SimMsg<S> {
    Terminal {
        sim_id: usize,
        path: Vec<PathStep>,
    },
    State {
        sim_id: usize,
        game_state: S,
        path: Vec<PathStep>,
        parent_node_id: NodeId,
        edge_index: usize,
    },
}

impl<S> SimMsg<S> {
    fn sim_id(&self) -> usize {
        match self {
            Self::Terminal { sim_id, .. } => *sim_id,
            Self::State { sim_id, .. } => *sim_id,
        }
    }
}
