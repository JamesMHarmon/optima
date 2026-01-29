use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::thread::JoinHandle;
use std::thread::spawn;
use crossbeam::channel::Receiver;
use crossbeam::channel::bounded;

use super::{NodeArena, NodeId};

const NUM_SELECTIONS: i32 = 10;
const BATCH_SIZE: usize = 32;


pub struct PUCT<'a, S, A, E, M, B, Sel> {
    game_engine: &'a E,
    analyzer: &'a M,
    backpropagation_strategy: &'a B,
    selection_strategy: &'a Sel,
    game_state: S,
    nodes: NodeArena<PUCTNode<A>>,
}

struct PUCTNode<A> {
    visits: usize,
    edges: Vec<PUCTEdge<A>>,
}

struct PUCTEdge<A> {
    action: A,
    prior: f32,
    visits: usize,
    child_idx: usize
}

impl<S, A, E, M, B, Sel> PUCT<'_, S, A, E, M, B, Sel> {
    pub fn search(&mut self) {
        let selection_workers = Self::run_selections(self);

        self.run_simulate();

        selection_workers.join();
    }

    fn run_simulate(&self) {
        self.nodes.get(NodeId::from_usize(10));
    }

    fn run_selections(&self) -> Workers {
        let (sender, receiver) = bounded(BATCH_SIZE);

        let mut handles = vec![];
        for _ in 0..NUM_SELECTIONS {
            let sender = sender.clone();
            let handle = spawn(move || {
                loop {
                    let selection = Self::select();
                    if sender.send(selection).is_err() {
                        break;
                    }
                }
            });
            handles.push(handle);
        }

        Workers { handles }
    }

    fn run_backpropagation(receiver: Receiver<usize>) {
        while let Ok(blee) = receiver.recv() {
            Self::backpropagate(blee);
        }
    }

    fn select() {

    }

    fn backpropagate(blee: usize) {

    }
}

struct Workers {
    handles: Vec<JoinHandle<()>>,
}

impl Workers {
    fn join(self) {
        for h in self.handles { h.join().unwrap(); }
    }
}

// Multi-thread implementationr
// Read: trace down the tree to find nodes to expand
    // - checks if the node is in a cache?
    // - put a node in the cache if not already
    // - have n number of threads

// Write: deterministic trace that updates node/edge values and backpropagate results up the tree
// Write: expands nodes
