use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread;

use common::TranspositionHash;
use crossbeam::channel::Receiver;
use dashmap::{DashMap, Entry};
use engine::GameEngine;
use model::GameAnalyzer;
use parking_lot::{Condvar, Mutex};

use crate::borrowed_or_owned::BorrowedOrOwned;
use crate::node::StateNode;
use crate::node_arena::NodeId;
use crate::node_graph_store::NodeGraphStore;
use crate::{RollupStats, SelectionPolicy, ValueModel};

const PROBE_THREADS: usize = 4;

type PuctStore<E, VM> = NodeGraphStore<<E as GameEngine>::Action, RollupOf<VM>>;

type RollupOf<VM> = <VM as ValueModel>::Rollup;

type SnapshotOf<VM> = <RollupOf<VM> as RollupStats>::Snapshot;

type EdgeKey = (NodeId, usize);

pub(super) struct Probe<'a, E, M, VM, Sel>
where
    E: GameEngine,
    VM: ValueModel,
{
    game_engine: &'a E,
    analyzer: &'a M,
    _value_model: &'a VM,
    selection_strategy: &'a Sel,
    store: &'a PuctStore<E, VM>,
    virtual_visit_map: DashMap<NodeId, Vec<u32>>,
    search_leaf_map: DashMap<u64, (u64, Vec<Vec<EdgeKey>>)>,
    num_probes: usize,
    probes_sent: AtomicU64,
    probes_reverted: AtomicU64,
    capacity_gate: CapacityGate,
    search_leaf_rx: Receiver<u64>,
}

impl<'a, E, M, VM, Sel> Probe<'a, E, M, VM, Sel>
where
    E: GameEngine + Sync,
    M: GameAnalyzer<State = E::State, Action = E::Action> + Sync,
    VM: ValueModel<State = E::State, Predictions = M::Predictions> + Sync,
    Sel: SelectionPolicy<SnapshotOf<VM>, State = E::State> + Sync,
    E::State: TranspositionHash + Sync,
    E::Action: Send + Sync,
    SnapshotOf<VM>: Send + Sync,
    RollupOf<VM>: Send + Sync,
{
    pub(super) fn new(
        game_engine: &'a E,
        analyzer: &'a M,
        _value_model: &'a VM,
        selection_strategy: &'a Sel,
        store: &'a PuctStore<E, VM>,
        search_leaf_rx: Receiver<u64>,
    ) -> Self {
        Self {
            game_engine,
            analyzer,
            _value_model,
            selection_strategy,
            search_leaf_rx,
            store,
            num_probes: 10000,
            probes_sent: AtomicU64::new(0),
            probes_reverted: AtomicU64::new(0),
            capacity_gate: CapacityGate::new(),
            virtual_visit_map: DashMap::new(),
            search_leaf_map: DashMap::new(),
        }
    }

    pub(super) fn run(&self, game_state: &E::State, alive: &AtomicBool) {
        thread::scope(|s| {
            let mut handles = Vec::with_capacity(PROBE_THREADS);

            for _ in 0..PROBE_THREADS {
                handles.push(s.spawn(|| self.run_probes(game_state, alive)));
            }

            self.process_leaf_completions();

            for handle in handles {
                handle.join().expect("Probe thread panicked");
            }
        });
    }

    fn run_probes(&self, game_state: &E::State, alive: &AtomicBool) {
        loop {
            if !alive.load(Ordering::Acquire) {
                break;
            } else if self.has_capacity() {
                self.probe_one(game_state);
            } else {
                self.wait_for_capacity(alive);
            }
        }
    }

    fn outstanding(&self) -> u64 {
        self.probes_sent
            .load(Ordering::Acquire)
            .saturating_sub(self.probes_reverted.load(Ordering::Acquire))
    }

    fn has_capacity(&self) -> bool {
        self.outstanding() < self.num_probes as u64
    }

    fn wait_for_capacity(&self, alive: &AtomicBool) {
        self.capacity_gate
            .wait_while(|| alive.load(Ordering::Acquire) && !self.has_capacity());
    }

    fn probe_one(&self, game_state: &E::State) {
        let selection = self.select_leaf(game_state);
        let new = self.store_search_leaf_path(selection.transposition_hash, selection.path);

        if !selection.is_terminal && new {
            self.analyzer.prefetch(&selection.game_state);
        }
    }

    fn process_leaf_completions(&self) {
        while let Ok(transposition_hash) = self.search_leaf_rx.recv() {
            if let Some((_, (epoch, paths))) = self.search_leaf_map.remove(&transposition_hash) {
                for path in paths {
                    for (node_id, edge_idx) in path {
                        self.virtual_visit_map.entry(node_id).and_modify(|v| {
                            if let Some(virtual_visits) = v.get_mut(edge_idx) {
                                *virtual_visits -= 1;
                            }
                        });
                    }
                }

                self.probes_reverted.fetch_max(epoch + 1, Ordering::Release);
                self.capacity_gate.notify_if(self.has_capacity());
            }
        }
    }

    fn select_leaf(&self, game_state: &E::State) -> SelectionResult<E::State> {
        let store = &self.store;
        let game_engine = &self.game_engine;
        let root_hash = game_state.transposition_hash();
        let mut current = self.store.get_node_id(root_hash).expect("No root node");
        let mut game_state = BorrowedOrOwned::Borrowed(game_state);
        let mut depth = 0;
        let mut path = Vec::new();

        loop {
            let node = store.state_node(current);
            let edge_idx = self.select_edge(&game_state, node, current, depth as u32);
            let (edge, action) = node.edge_and_action(edge_idx);

            let next_game_state = game_engine.take_action(&game_state, action);
            let terminal_state = game_engine.terminal_state(&next_game_state);
            let transposition_hash = next_game_state.transposition_hash();
            let is_terminal = terminal_state.is_some();

            depth += 1;
            self.increment_virtual_visit(current, edge_idx);
            path.push((current, edge_idx));

            if is_terminal {
                return SelectionResult::terminal(next_game_state, transposition_hash, path);
            }

            if let Some(node_id) = self.store.get_state_node_id(edge, transposition_hash) {
                game_state = BorrowedOrOwned::Owned(next_game_state);
                current = node_id;
                continue;
            }

            return SelectionResult::unexpanded(next_game_state, transposition_hash, path);
        }
    }

    fn store_search_leaf_path(&self, transposition_hash: u64, path: Vec<(NodeId, usize)>) -> bool {
        match self.search_leaf_map.entry(transposition_hash) {
            Entry::Vacant(entry) => {
                let epoch = self.probes_sent.fetch_add(1, Ordering::Release);
                entry.insert((epoch, vec![path]));
                true
            }
            Entry::Occupied(mut entry) => {
                entry.get_mut().1.push(path);
                false
            }
        }
    }

    fn select_edge(
        &self,
        game_state: &E::State,
        node: &StateNode<E::Action, RollupOf<VM>>,
        node_id: NodeId,
        depth: u32,
    ) -> usize {
        node.ensure_frontier_edge();

        let vv_map_entry = self.virtual_visit_map.get(&node_id);
        let virtual_visits_map = vv_map_entry.map(|v| v.clone()).unwrap_or_default();

        let virtual_visits_sum = virtual_visits_map.iter().sum::<u32>();
        let node_visits = node.visits() + virtual_visits_sum;

        let edge_iter = self.store.iter_edge_info(node).map(|mut edge_info| {
            let edge_index = edge_info.edge_index;
            let virtual_visits = virtual_visits_map.get(edge_index).copied().unwrap_or(0);

            edge_info.visits += virtual_visits;

            edge_info
        });

        self.selection_strategy
            .select_edge(edge_iter, node_visits, game_state, depth)
    }

    fn increment_virtual_visit(&self, node_id: NodeId, edge_idx: usize) {
        let mut v = self.virtual_visit_map.entry(node_id).or_default();
        if v.len() <= edge_idx {
            v.resize(edge_idx + 1, 0);
        }
        v[edge_idx] += 1;
    }
}

struct SelectionResult<S> {
    game_state: S,
    transposition_hash: u64,
    is_terminal: bool,
    path: Vec<(NodeId, usize)>,
}

impl<S> SelectionResult<S> {
    fn terminal(game_state: S, transposition_hash: u64, path: Vec<(NodeId, usize)>) -> Self {
        Self {
            game_state,
            transposition_hash,
            is_terminal: true,
            path,
        }
    }

    fn unexpanded(game_state: S, transposition_hash: u64, path: Vec<(NodeId, usize)>) -> Self {
        Self {
            game_state,
            transposition_hash,
            is_terminal: false,
            path,
        }
    }
}

/// A condition variable gate that allows threads to wait until a condition
/// is met and be notified when it becomes true.
struct CapacityGate {
    lock: Mutex<()>,
    cvar: Condvar,
}

impl CapacityGate {
    fn new() -> Self {
        Self {
            lock: Mutex::new(()),
            cvar: Condvar::new(),
        }
    }

    /// Acquires the lock and sleeps while `should_wait()` returns true.
    /// The predicate is re-evaluated under the lock to prevent missed wakeups.
    fn wait_while(&self, should_wait: impl Fn() -> bool) {
        let mut guard = self.lock.lock();
        if should_wait() {
            self.cvar.wait(&mut guard);
        }
    }

    /// Wakes all waiting threads if `condition` is true.
    /// Acquires the lock before notifying to prevent missed wakeups.
    fn notify_if(&self, condition: bool) {
        if condition {
            let _guard = self.lock.lock();
            self.cvar.notify_all();
        }
    }
}
