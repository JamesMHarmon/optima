use std::collections::{BTreeMap, HashMap};

use crossbeam::channel::{Receiver, TryRecvError};

use crate::node_arena::NodeId;
use crate::node_graph_store::NodeGraphStore;
use crate::rollup::RollupStats;
use crate::simulate::WaiterInfo;
use crate::value_model::ValueModel;
use common::TranspositionHash;
use model::{ActionWithPolicy, GameAnalyzer};

type PuctStore<M, R> = NodeGraphStore<<M as GameAnalyzer>::Action, R>;
type RollupOf<VM> = <VM as ValueModel>::Rollup;
type SimRx<S> = Receiver<SimMsg<S>>;

/// Owns the receive-and-backprop loop; intended to run on a dedicated thread.
///
/// Receives [`SimMsg`]s from the simulation thread, collects network results
/// via [`GameAnalyzer::recv`], and drains [`BackpropQueue`] in strict
/// simulation-ID order to guarantee unbiased backpropagation.
pub(super) struct Backpropagator<'a, M, VM>
where
    M: GameAnalyzer,
    VM: ValueModel,
{
    analyzer: &'a M,
    store: &'a PuctStore<M, RollupOf<VM>>,
    value_model: &'a VM,
}

impl<'a, M, VM> Backpropagator<'a, M, VM>
where
    M: GameAnalyzer<RequestId = usize>,
    VM: ValueModel<State = M::State, Predictions = M::Predictions>,
    RollupOf<VM>: RollupStats,
    M::State: TranspositionHash + Clone,
    M::Action: Send + Sync,
{
    pub(super) fn new(
        analyzer: &'a M,
        store: &'a PuctStore<M, RollupOf<VM>>,
        value_model: &'a VM,
    ) -> Self {
        Self {
            analyzer,
            store,
            value_model,
        }
    }

    pub(super) fn run(&self, rx: SimRx<M::State>) {
        let mut backprop_queue = BackpropQueue::new();
        // Maps transposition hash → (game_state, pending waiters).
        let mut expanding: HashMap<u64, (M::State, Vec<WaiterInfo>)> = HashMap::new();
        // Maps request_id → transposition hash for reverse lookup on recv.
        let mut request_to_hash: HashMap<usize, u64> = HashMap::new();
        // Remembers the NodeId created for each resolved hash so that late
        // Waiter messages (racing with resolution) can still form their task.
        let mut resolved: HashMap<u64, NodeId> = HashMap::new();
        let mut sim_done = false;

        loop {
            // Drain all immediately available messages from the sim thread.
            loop {
                match rx.try_recv() {
                    Ok(SimMsg::Terminal { sim_id, task }) => {
                        backprop_queue.push(sim_id, task);
                    }
                    Ok(SimMsg::NewLeaf {
                        request_id,
                        hash,
                        game_state,
                        waiter,
                    }) => {
                        request_to_hash.insert(request_id, hash);
                        expanding.insert(hash, (game_state, vec![waiter]));
                    }
                    Ok(SimMsg::Waiter { hash, waiter }) => {
                        if let Some((_, waiters)) = expanding.get_mut(&hash) {
                            // Expansion still in flight: queue with the rest.
                            waiters.push(waiter);
                        } else if let Some(&new_node_id) = resolved.get(&hash) {
                            // Late waiter: expansion already completed.
                            backprop_queue.push(
                                waiter.sim_id,
                                BackpropTask {
                                    path: waiter.path,
                                    parent_node_id: waiter.parent_node_id,
                                    edge_index: waiter.edge_index,
                                    new_node_id: Some(new_node_id),
                                },
                            );
                        }
                    }
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => {
                        sim_done = true;
                        break;
                    }
                }
            }

            // Block on one network result while there are outstanding requests.
            if !request_to_hash.is_empty() {
                let (request_id, analysis) = self.analyzer.recv();
                if let Some(hash) = request_to_hash.remove(&request_id)
                    && let Some((game_state, waiters)) = expanding.remove(&hash)
                {
                    let (policy_priors, predictions) = analysis.into_inner();
                    let new_node_id: NodeId =
                        self.create_state_node(hash, policy_priors, &game_state, &predictions);
                    resolved.insert(hash, new_node_id);
                    for waiter in waiters {
                        backprop_queue.push(
                            waiter.sim_id,
                            BackpropTask {
                                path: waiter.path,
                                parent_node_id: waiter.parent_node_id,
                                edge_index: waiter.edge_index,
                                new_node_id: Some(new_node_id),
                            },
                        );
                    }
                }
            }

            // Drain backprop tasks in strict simulation-ID order.
            let store = self.store;
            backprop_queue.drain_ready(|task| {
                if let Some(new_id) = task.new_node_id {
                    let parent = store.state_node(task.parent_node_id);
                    let (edge, _) = parent.edge_and_action(task.edge_index);
                    store.graph().add_child_to_edge(edge, new_id);
                }
                for &node_id in task.path.iter().rev() {
                    store.recompute_rollup(node_id);
                }
            });

            if sim_done && request_to_hash.is_empty() && !backprop_queue.has_pending() {
                break;
            }
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

/// Accumulates backprop tasks and releases them in strict simulation-ID order.
///
/// Guarantees that terminal results never "skip ahead" of in-flight network
/// expansions, eliminating the latency-induced visit-count bias that arises
/// when fast paths are consistently propagated before slow ones.
pub(super) struct BackpropQueue {
    pending: BTreeMap<usize, BackpropTask>,
    next_id: usize,
}

impl BackpropQueue {
    pub(super) fn new() -> Self {
        Self {
            pending: BTreeMap::new(),
            next_id: 0,
        }
    }

    /// Enqueue a task to be processed when simulation `sim_id` is next in line.
    pub(super) fn push(&mut self, sim_id: usize, task: BackpropTask) {
        self.pending.insert(sim_id, task);
    }

    /// Call `f` for every consecutively-ready task (starting from the current
    /// `next_id`), advancing the counter after each call.
    pub(super) fn drain_ready(&mut self, mut f: impl FnMut(BackpropTask)) {
        while let Some(task) = self.pending.remove(&self.next_id) {
            f(task);
            self.next_id += 1;
        }
    }

    pub(super) fn has_pending(&self) -> bool {
        !self.pending.is_empty()
    }
}

/// All data needed to complete one backpropagation step.
pub(super) struct BackpropTask {
    pub(super) path: Vec<NodeId>,
    pub(super) parent_node_id: NodeId,
    pub(super) edge_index: usize,
    /// When `Some`, this new node must be linked to the parent edge before backpropagating.
    pub(super) new_node_id: Option<NodeId>,
}

/// Message sent from the simulation thread to the backprop thread.
pub(super) enum SimMsg<S> {
    /// A terminal leaf was reached; the backprop task is fully formed.
    Terminal { sim_id: usize, task: BackpropTask },
    /// A new (not yet analysed) leaf was found. `analyzer.analyze` has already
    /// been called before this message is sent.
    NewLeaf {
        request_id: usize,
        hash: u64,
        game_state: S,
        waiter: WaiterInfo,
    },
    /// A leaf whose transposition hash is already in-flight; append to waiters.
    Waiter { hash: u64, waiter: WaiterInfo },
}
