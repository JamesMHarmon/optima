use std::collections::BTreeMap;

use crossbeam::channel::Receiver;

use crate::analysis_coordinator::InFlightExpansions;
use crate::node_arena::NodeId;
use crate::node_graph_store::NodeGraphStore;
use crate::rollup::RollupStats;
use crate::value_model::ValueModel;
use common::TranspositionHash;
use model::{ActionWithPolicy, GameAnalyzer, GameStateAnalysis};

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

    analysis_buffer: BTreeMap<usize, GameStateAnalysis<M::Action, M::Predictions>>,
    pending_by_sim_id: BTreeMap<usize, PendingSim<M::State>>,
    next_sim_id: usize,
    sim_done: bool,
}

impl<'a, M, VM, IE> Backpropagator<'a, M, VM, IE>
where
    M: GameAnalyzer<RequestId = usize>,
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

            analysis_buffer: BTreeMap::new(),
            pending_by_sim_id: BTreeMap::new(),
            next_sim_id: 0,
            sim_done: false,
        }
    }

    pub(super) fn run(mut self) {
        while let Some(pending) = self.next_sim() {
            match pending {
                PendingSim::Terminal(p) => {
                    if let Some(link) = p.link {
                        self.link_child(link.parent_node_id, link.edge_index, link.new_node_id);
                    }
                    self.backprop_path(p.path);
                }
                PendingSim::State(p) => {
                    let analysis = self.recv_analysis_for(p.request_id);

                    let new_node_id = if let Some(existing) = self.store.get_node_id(p.hash) {
                        existing
                    } else {
                        let (policy_priors, predictions) = analysis.into_inner();
                        self.create_state_node(p.hash, policy_priors, &p.game_state, &predictions)
                    };

                    self.expansions.complete(p.hash);

                    self.link_child(p.parent_node_id, p.edge_index, new_node_id);
                    self.backprop_path(p.path);
                }
            }
        }
    }

    fn next_sim(&mut self) -> Option<PendingSim<M::State>> {
        loop {
            if let Some(pending) = self.pending_by_sim_id.remove(&self.next_sim_id) {
                self.next_sim_id += 1;
                return Some(pending);
            }

            if self.sim_done {
                if self.pending_by_sim_id.is_empty() {
                    return None;
                }

                // Should be unreachable in normal operation: sim ids are expected
                // to be contiguous from 0. If the channel is closed early (or a
                // bug drops a sim id), don't stall forever.
                let min_id = *self
                    .pending_by_sim_id
                    .keys()
                    .next()
                    .expect("pending_by_sim_id is non-empty");
                self.next_sim_id = min_id;
                continue;
            }

            match self.rx.recv() {
                Ok(msg) => {
                    let sim_id = msg.sim_id();
                    self.pending_by_sim_id
                        .insert(sim_id, PendingSim::from_msg(msg));
                }
                Err(_) => self.sim_done = true,
            }
        }
    }

    fn recv_analysis_for(
        &mut self,
        request_id: usize,
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

    fn backprop_path(&self, path: Vec<NodeId>) {
        let store = self.store;
        for &node_id in path.iter().rev() {
            store.recompute_rollup(node_id);
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
            SimMsg::Terminal { path, link, .. } => Self::Terminal(PendingTerminal { path, link }),
            SimMsg::State {
                request_id,
                hash,
                game_state,
                path,
                parent_node_id,
                edge_index,
                ..
            } => Self::State(PendingState {
                request_id,
                hash,
                game_state,
                path,
                parent_node_id,
                edge_index,
            }),
        }
    }
}

struct PendingTerminal {
    path: Vec<NodeId>,
    link: Option<LinkChild>,
}

struct PendingState<S> {
    request_id: usize,
    hash: u64,
    game_state: S,
    path: Vec<NodeId>,
    parent_node_id: NodeId,
    edge_index: usize,
}

pub(super) struct LinkChild {
    parent_node_id: NodeId,
    edge_index: usize,
    new_node_id: NodeId,
}

impl LinkChild {
    pub(crate) fn new(parent_node_id: NodeId, edge_index: usize, new_node_id: NodeId) -> Self {
        Self {
            parent_node_id,
            edge_index,
            new_node_id,
        }
    }
}

/// Message sent from the simulation thread to the backprop thread.
pub(super) enum SimMsg<S> {
    Terminal {
        sim_id: usize,
        path: Vec<NodeId>,
        /// When `Some`, link this newly-created child before backpropagating.
        link: Option<LinkChild>,
    },
    /// A new (not yet analysed) leaf was found. `analyzer.analyze` has already
    /// been called before this message is sent.
    State {
        sim_id: usize,
        request_id: usize,
        hash: u64,
        game_state: S,
        path: Vec<NodeId>,
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
