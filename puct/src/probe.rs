use std::sync::atomic::{AtomicBool, AtomicU32};

use crossbeam::channel::Receiver;

pub(super) struct Probe<'a, E, M, VM, Sel> {
    game_engine: &'a E,
    analyzer: &'a M,
    value_model: &'a VM,
    selection_strategy: &'a Sel,
    num_probes: usize,
    num_outstanding: AtomicU32,
    search_leaf_rx: Receiver<u64>,
}

impl<'a, E, M, VM, Sel> Probe<'a, E, M, VM, Sel> {
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

    pub(super) fn run(&self, alive: &AtomicBool) {}
}
