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
        let root_node_id = NodeId::from_usize(0);
        let selection = self.select_leaf(root_node_id);
        let expansion = self.evaluate_leaf(&selection);
        self.apply_expansion(expansion);
    }

    fn select_leaf(&self, node_id: NodeId) -> SelectionResult<S> {
        let mut path = vec![];
        let mut current = node_id;
        let mut state = self.game_state.clone();
        
        loop {
            let node = self.nodes.get(current);
            path.push(current);
            
            if node.is_unexpanded() {
                break;
            }
            
            let edge_idx = self.select_best_edge(node);
            let edge = &node.edges[edge_idx];
            
            current = edge.child;
            state = self.game_engine.take_action(&state, &edge.action);
        }
        
        SelectionResult { path, leaf_state: state }
    }

    fn evaluate_leaf(&self, selection: &SelectionResult<S>) -> ExpansionResult<A, P, R> {
        let analysis = self.analyzer.get_state_analysis(&selection.leaf_state);
        let new_node = self.create_node_from_analysis(analysis);
        let value = self.evaluate_node(&new_node);
        
        ExpansionResult {
            path: selection.path,
            new_node,
            value,
        }
    }

    fn apply_expansion(&mut self, result: ExpansionResult<A, P, R>) {
        let new_node_id = self.nodes.push(result.new_node);
        
        if let Some(&parent_id) = result.path.last() {
            let parent = self.nodes.get_mut(parent_id);
        }
        
        for &node_id in result.path.iter().rev() {
            let node = self.nodes.get_mut(node_id);
            node.visits += 1;
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


struct SelectionResult<S> {
    path: Vec<NodeId>,
    leaf_state: S,
}

struct ExpansionResult< A, P, R> {
    path: Vec<NodeId>,
    new_node: PUCTNode<A, P, R>,
    value: f32,
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
