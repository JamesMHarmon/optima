use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::thread;

use common::TranspositionHash;
use crossbeam::channel::Receiver;
use dashmap::{DashMap, Entry};
use engine::GameEngine;
use model::GameAnalyzer;

use crate::borrowed_or_owned::BorrowedOrOwned;
use crate::node::StateNode;
use crate::node_arena::NodeId;
use crate::node_graph_store::NodeGraphStore;
use crate::{RollupStats, SelectionPolicy, ValueModel};

const PROBE_THREADS: usize = 2;

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
    search_leaf_map: DashMap<u64, Vec<Vec<EdgeKey>>>,
    num_probes: usize,
    num_outstanding: AtomicU32,
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
            num_probes: 2048,
            num_outstanding: AtomicU32::new(0),
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

            self.revert_virtual_visits();

            for handle in handles {
                handle.join().expect("Probe thread panicked");
            }
        });
    }

    fn run_probes(&self, game_state: &E::State, alive: &AtomicBool) {
        while alive.load(Ordering::Acquire) {
            self.probe_one(game_state);
        }
    }

    fn probe_one(&self, game_state: &E::State) {
        let selection = self.select_leaf(game_state);
        let new = self.store_search_leaf_path(selection.transposition_hash, selection.path);

        if !selection.is_terminal && new {
            self.analyzer.prefetch(&selection.game_state);
        }
    }

    fn revert_virtual_visits(&self) {
        while let Ok(transposition_hash) = self.search_leaf_rx.recv() {
            if let Some(paths) = self.search_leaf_map.remove(&transposition_hash) {
                for path in paths.1 {
                    for (node_id, edge_idx) in path {
                        self.virtual_visit_map.entry(node_id).and_modify(|v| {
                            if let Some(virtual_visits) = v.get_mut(edge_idx) {
                                *virtual_visits -= 1;
                            }
                        });
                    }
                }
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
        let entry = self.search_leaf_map.entry(transposition_hash);
        let vacant = matches!(entry, Entry::Vacant(_));
        entry.or_default().push(path);
        vacant
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
        self.virtual_visit_map
            .entry(node_id)
            .and_modify(|v| {
                if v.len() <= edge_idx {
                    v.resize(edge_idx + 1, 0);
                }
                v[edge_idx] += 1;
            })
            .or_insert_with(|| {
                let mut vec = vec![0; edge_idx + 1];
                vec[edge_idx] = 1;
                vec
            });
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
