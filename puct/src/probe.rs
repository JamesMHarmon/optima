use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::thread;

use crossbeam::channel::Receiver;
use engine::GameEngine;
use model::GameAnalyzer;

use crate::{RollupStats, SelectionPolicy, ValueModel};

const PROBE_THREADS: usize = 2;

type RollupOf<VM> = <VM as ValueModel>::Rollup;

type SnapshotOf<VM> = <RollupOf<VM> as RollupStats>::Snapshot;

pub(super) struct Probe<'a, E, M, VM, Sel> {
    game_engine: &'a E,
    analyzer: &'a M,
    value_model: &'a VM,
    selection_strategy: &'a Sel,
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
{
    pub(super) fn new(
        game_engine: &'a E,
        analyzer: &'a M,
        value_model: &'a VM,
        selection_strategy: &'a Sel,
        search_leaf_rx: Receiver<u64>,
    ) -> Self {
        Self {
            game_engine,
            analyzer,
            value_model,
            selection_strategy,
            num_probes: 0,
            num_outstanding: AtomicU32::new(0),
            search_leaf_rx,
        }
    }

    pub(super) fn run(&self, alive: &AtomicBool) {
        thread::scope(|s| {
            let mut handles = Vec::with_capacity(PROBE_THREADS + 1);

            for _ in 0..PROBE_THREADS {
                let handle = s.spawn(|| self.run_probe(alive));
                handles.push(handle);
            }

            self.run_cleanup();

            for handle in handles {
                handle.join().expect("Probe thread panicked");
            }
        });
    }

    fn run_probe(&self, alive: &AtomicBool) {
        while alive.load(Ordering::Relaxed) {}
    }

    fn run_cleanup(&self) {}
}
