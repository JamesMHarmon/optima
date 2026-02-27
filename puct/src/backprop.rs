use std::collections::{BTreeMap, HashMap};

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
    pending_sims: BTreeMap<usize, PendingSim<M::State>>,
    next_sim_id: usize,
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
            pending_sims: BTreeMap::new(),
            next_sim_id: 0,
        }
    }

    pub(super) fn run(mut self) {
        while let Some(pending) = self.next_sim() {
            let mut leaf_child_id: Option<NodeId> = None;

            if let PendingSim::State(p) = &pending {
                let game_state = &p.game_state;
                let hash = p.game_state.transposition_hash();

                self.expansions.complete(hash);

                if self.store.get_node_id(hash).is_some() {
                    // Another simulation has already expanded this node and backpropagated the path, so we can skip it.
                    self.remove_virtual_loss(pending.path());
                    continue;
                }

                let analysis = self.recv_analysis_for(hash);

                let (policy_priors, predictions) = analysis.into_inner();
                let new_node_id =
                    self.create_state_node(hash, policy_priors, game_state, &predictions);
                self.link_child(p.parent_node_id, p.edge_index, new_node_id);

                leaf_child_id = Some(new_node_id);
            }

            self.commit_sim(&pending, leaf_child_id);
        }
    }

    fn commit_sim(&self, sim: &PendingSim<M::State>, leaf_child_id: Option<NodeId>) {
        let store = self.store;
        let graph = store.graph();
        let path = sim.path();

        if path.is_empty() {
            return;
        }

        // Track the "child" reached by the current step so we can increment
        // afterstate outcome visits when the edge points to an AfterState.
        let mut child_for_outcome: Option<NodeId> = match sim {
            PendingSim::State(_) => leaf_child_id,
            PendingSim::Terminal(_) => {
                let last = path.last().expect("path not empty");
                let last_node = store.state_node(last.node_id);
                let last_edge = last_node.edge(last.edge_index);
                graph.find_edge_terminal(last_edge)
            }
        };

        for step in path.iter().rev() {
            let node = store.state_node(step.node_id);
            let edge = node.edge(step.edge_index);

            if let Some(child_id) = child_for_outcome {
                graph.increment_afterstate_outcome_visits(edge, child_id);
            }

            node.increment_visits();
            edge.increment_visits();

            store.recompute_rollup(step.node_id);

            node.decrement_virtual_visits();
            edge.decrement_virtual_visits();

            child_for_outcome = Some(step.node_id);
        }
    }

    fn next_sim(&mut self) -> Option<PendingSim<M::State>> {
        // Check the BTreeMap to see if the sim has already been received.
        if let Some(sim) = self.try_next_sim() {
            return Some(sim);
        }

        // The next sim is not in the BTreeMap so we receive messages until we find it.
        while let Ok(sim) = self.rx.recv() {
            let sim_id = sim.sim_id();

            if sim_id == self.next_sim_id {
                self.next_sim_id += 1;
                return Some(PendingSim::from_msg(sim));
            }

            self.pending_sims.insert(sim_id, PendingSim::from_msg(sim));
        }

        // No more messages will arive and we have processed all pending messages in order, so we're done.
        if self.pending_sims.is_empty() {
            return None;
        }

        panic!(
            "Missing simulation(s): expected sim_id {}",
            self.next_sim_id
        );
    }

    fn try_next_sim(&mut self) -> Option<PendingSim<M::State>> {
        if let Some((&min_id, _)) = self.pending_sims.first_key_value()
            && min_id == self.next_sim_id
        {
            let (_, pending) = self.pending_sims.pop_first()?;
            self.next_sim_id += 1;
            return Some(pending);
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

#[cfg(test)]
mod tests;
