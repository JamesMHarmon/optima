use crossbeam::channel::Receiver;
use crossbeam::channel::bounded;
use half::f16;
use model::Analyzer;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::thread::JoinHandle;
use std::thread::spawn;

use super::{NodeArena, NodeId};

const NUM_SELECTIONS: i32 = 10;
const BATCH_SIZE: usize = 32;

pub struct PUCT<'a, S, A, M, E, B, Sel, P, R>
where
    M: Analyzer,
    B: BackpropagationStrategy,
{
    game_engine: &'a E,
    analyzer: &'a M,
    backpropagation_strategy: &'a B,
    selection_strategy: &'a Sel,
    game_state: S,
    nodes: NodeArena<PUCTNode<A, M::Predictions, B::RollupStats>>,
}

struct PUCTNode<A, P, R> {
    visits: u32,
    predictions: P,
    rollup_stats: R,
    edges: Vec<PUCTEdge<A>>,
}

struct PUCTEdge<A> {
    action: A,
    policy_prior: f16,
    visits: u32,
    child: NodeId,
}

impl<A, P, R> PUCTNode<A, P, R> {
    fn is_unexpanded(&self) -> bool {
        self.visits == 0 || self.edges.is_empty()
    }
}

impl<S, A, M, E, B, Sel, P, R> PUCT<'_, S, A, M, E, B, Sel, P, R> {
    pub fn search(&mut self) {
        let selection_workers = Self::run_selections(self);

        self.run_simulate();

        selection_workers.join();
    }

    fn run_simulate(&self) {
        let current_node = self.get_root_node();

        loop {
            // Check if terminal or unexpanded
            if current_node.is_unexpanded() {
                break;
            }

            // TODO: select best edge and move to child
            break;
        }
    }

    fn get_root_node(&self) -> &PUCTNode<A, P, R> {
        self.nodes.get(NodeId::from_usize(0))
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

    fn select() {}

    fn backpropagate(blee: usize) {}
}

struct Workers {
    handles: Vec<JoinHandle<()>>,
}

impl Workers {
    fn join(self) {
        for h in self.handles {
            h.join().unwrap();
        }
    }
}

// Multi-thread implementationr
// Read: trace down the tree to find nodes to expand
// - checks if the node is in a cache?
// - put a node in the cache if not already
// - have n number of threads

// Write: deterministic trace that updates node/edge values and backpropagate results up the tree
// Write: expands nodes
