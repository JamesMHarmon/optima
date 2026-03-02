use std::collections::{BTreeMap, HashMap};

use crossbeam::channel::Receiver;

use crate::analysis_coordinator::InFlightExpansions;
use crate::edge::PUCTEdge;
use crate::node_arena::NodeId;
use crate::node_graph_store::NodeGraphStore;
use crate::rollup::RollupStats;
use crate::search_context::PathStep;
use crate::value_model::ValueModel;
use common::TranspositionHash;
use model::{GameAnalyzer, GameStateAnalysis, RequestId};

type PuctStore<M, R> = NodeGraphStore<<M as GameAnalyzer>::Action, R>;
type RollupOf<VM> = <VM as ValueModel>::Rollup;
type SnapshotOf<VM> = <VM as ValueModel>::Snapshot;
type SimRx<S, TS> = Receiver<SimMsg<S, TS>>;

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
    rx: SimRx<M::State, SnapshotOf<VM>>,

    analysis_buffer: HashMap<RequestId, GameStateAnalysis<M::Action, M::Predictions>>,
    pending_sims: BTreeMap<usize, SimMsg<M::State, SnapshotOf<VM>>>,
    next_sim_id: usize,
}

impl<'a, M, VM, IE> Backpropagator<'a, M, VM, IE>
where
    M: GameAnalyzer,
    VM: ValueModel<Predictions = M::Predictions>,
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
        rx: SimRx<M::State, SnapshotOf<VM>>,
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
        while let Some(sim) = self.next_sim() {
            match sim {
                SimMsg::Terminal(sim) => {
                    let last_step = sim.last_step();
                    let terminal_node_id = self.upsert_terminal(last_step, &sim.terminal_snapshot);
                    self.commit_path(&sim.path, terminal_node_id);
                }
                SimMsg::State(sim) => {
                    let store = self.store;
                    let last_step = sim.last_step();

                    let hash = sim.game_state.transposition_hash();
                    self.expansions.complete(hash);

                    // Check if another simulation has already expanded this node and backpropagated the path, so we can skip it.
                    if store.get_node_id(hash).is_some() {
                        self.remove_virtual_loss(&sim.path);
                        continue;
                    }

                    let analysis = self.recv_analysis_for(hash);
                    let (priors, preds) = analysis.into_inner();

                    let snapshot = self.value_model.pred_snapshot(&preds);
                    let rollup = snapshot.into();

                    let new_node_id = store.create_and_insert_state_node(hash, priors, rollup);
                    self.link_state_node(last_step.node_id, last_step.edge_index, new_node_id);

                    self.commit_path(&sim.path, new_node_id);
                }
            }
        }
    }

    fn commit_path(&self, path: &[PathStep], leaf_child_id: NodeId) {
        let store = self.store;
        let graph = store.graph();

        if path.is_empty() {
            return;
        }

        // Track the "child" reached by the current step so we can increment
        // afterstate outcome visits when the edge points to an AfterState.
        let mut child_for_outcome = leaf_child_id;

        for step in path.iter().rev() {
            let node = store.state_node(step.node_id);
            let edge = node.edge(step.edge_index);

            node.increment_visits();
            edge.increment_visits();
            graph.increment_afterstate_outcome_visits(edge, child_for_outcome);

            store.recompute_rollup(step.node_id);

            node.decrement_virtual_visits();
            edge.decrement_virtual_visits();

            child_for_outcome = step.node_id;
        }
    }

    fn upsert_terminal(&self, last_step: &PathStep, terminal_snapshot: &SnapshotOf<VM>) -> NodeId {
        let store = self.store;

        let last_node = store.state_node(last_step.node_id);
        let last_edge = last_node.edge(last_step.edge_index);

        self.upsert_terminal_edge(last_edge, terminal_snapshot)
    }

    fn upsert_terminal_edge(&self, edge: &PUCTEdge, terminal_snapshot: &SnapshotOf<VM>) -> NodeId {
        let graph = self.store.graph();

        if let Some(terminal_id) = graph.find_edge_terminal(edge) {
            let terminal_node = self.store.terminal_node(terminal_id);
            terminal_node.rollup_stats().accumulate(terminal_snapshot);
            terminal_id
        } else {
            let rollup_stats: RollupOf<VM> = (*terminal_snapshot).into();
            let terminal_id = self.store.create_and_insert_terminal_node(rollup_stats);
            graph.add_child_to_edge(edge, terminal_id);
            terminal_id
        }
    }

    fn next_sim(&mut self) -> Option<SimMsg<M::State, SnapshotOf<VM>>> {
        // Check the BTreeMap to see if the sim has already been received.
        if let Some(sim) = self.try_next_sim() {
            return Some(sim);
        }

        // The next sim is not in the BTreeMap so we receive messages until we find it.
        while let Ok(sim) = self.rx.recv() {
            let sim_id = sim.sim_id();

            if sim_id == self.next_sim_id {
                self.next_sim_id += 1;
                return Some(sim);
            }

            self.pending_sims.insert(sim_id, sim);
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

    fn try_next_sim(&mut self) -> Option<SimMsg<M::State, SnapshotOf<VM>>> {
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

    fn link_state_node(&self, parent_node_id: NodeId, edge_index: usize, new_node_id: NodeId) {
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
}

/// Message sent from the simulation thread to the backprop thread.
pub(super) struct TerminalSimMsg<TS> {
    pub(super) sim_id: usize,
    pub(super) path: Vec<PathStep>,
    pub(super) terminal_snapshot: TS,
}

/// Message sent from the simulation thread to the backprop thread.
pub(super) struct StateSimMsg<S> {
    pub(super) sim_id: usize,
    pub(super) game_state: S,
    pub(super) path: Vec<PathStep>,
}

/// Message sent from the simulation thread to the backprop thread.
pub(super) enum SimMsg<S, TS> {
    Terminal(TerminalSimMsg<TS>),
    State(StateSimMsg<S>),
}

impl<S, TS> SimMsg<S, TS> {
    pub(super) fn new_terminal(sim_id: usize, path: Vec<PathStep>, terminal_snapshot: TS) -> Self {
        SimMsg::Terminal(TerminalSimMsg {
            sim_id,
            path,
            terminal_snapshot,
        })
    }

    pub(super) fn new_state(sim_id: usize, game_state: S, path: Vec<PathStep>) -> Self {
        SimMsg::State(StateSimMsg {
            sim_id,
            game_state,
            path,
        })
    }

    fn sim_id(&self) -> usize {
        match self {
            Self::Terminal(msg) => msg.sim_id,
            Self::State(msg) => msg.sim_id,
        }
    }
}

impl<TS> TerminalSimMsg<TS> {
    fn last_step(&self) -> &PathStep {
        self.path.last().expect("Must contain at least one step")
    }
}

impl<S> StateSimMsg<S> {
    fn last_step(&self) -> &PathStep {
        self.path.last().expect("Must contain at least one step")
    }
}

#[cfg(test)]
mod tests;
