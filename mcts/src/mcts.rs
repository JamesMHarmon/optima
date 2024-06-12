use crate::{node, BackpropagationStrategy, SelectionStrategy};

use super::{DirichletOptions, MCTSEdge, MCTSNode, MCTSOptions, NodeDetails};
use super::{Temperature, PUCT};
use anyhow::{anyhow, Context, Result};
use common::div_or_zero;
use engine::{GameEngine, GameState};
use futures::stream::{FuturesUnordered, StreamExt};
use generational_arena::{Arena, Index};
use itertools::Itertools;
use log::warn;
use model::EdgeMetrics;
use model::{GameAnalyzer, NodeMetrics};
use rand::distributions::WeightedIndex;
use rand::prelude::Distribution;
use rand::{thread_rng, Rng};
use rand_distr::Dirichlet;
use std::borrow::Cow;
use std::cell::{Ref, RefCell, RefMut};
use std::fmt::Debug;
use std::ops::Deref;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

pub struct MCTS<'a, S, A, E, M, B, Sel, P, T, PV> {
    options: MCTSOptions,
    game_engine: &'a E,
    analyzer: &'a M,
    backpropagation_strategy: B,
    selection_strategy: Sel,
    game_state: S,
    root: Option<Index>,
    arena: NodeArena<MCTSNode<A, P, PV>>,
    focus_actions: Vec<A>,
    temp: T,
}

#[allow(non_snake_case)]
impl<'a, S, A, E, M, B, Sel, P, T, PV> MCTS<'a, S, A, E, M, B, Sel, P, T, PV>
where
    S: GameState,
    A: Clone + Eq + Debug,
    E: 'a + GameEngine<State = S, Action = A, Terminal = P>,
    M: 'a + GameAnalyzer<State = S, Action = A, Predictions = P>,
    T: Temperature<State = S>,
    PV: Default,
{
    pub fn new(
        game_state: S,
        game_engine: &'a E,
        analyzer: &'a M,
        backpropagation_strategy: B,
        selection_strategy: Sel,
        options: MCTSOptions,
        temp: T,
    ) -> Self {
        MCTS {
            options,
            game_engine,
            analyzer,
            backpropagation_strategy,
            selection_strategy,
            game_state,
            root: None,
            arena: NodeArena::with_capacity(800 * 2),
            focus_actions: vec![],
            temp,
        }
    }

    pub fn with_capacity(
        game_state: S,
        game_engine: &'a E,
        analyzer: &'a M,
        backpropagation_strategy: B,
        selection_strategy: Sel,
        options: MCTSOptions,
        capacity: usize,
        temp: T,
    ) -> Self {
        MCTS {
            options,
            game_engine,
            analyzer,
            backpropagation_strategy,
            selection_strategy,
            game_state,
            root: None,
            arena: NodeArena::with_capacity(capacity * 2),
            focus_actions: vec![],
            temp,
        }
    }

    pub fn select_action(&mut self) -> Result<A> {
        self._select_action(true)
    }

    pub fn select_action_no_temp(&mut self) -> Result<A> {
        self._select_action(false)
    }

    pub async fn advance_to_action(&mut self, action: A) -> Result<()> {
        self.advance_to_action_clearable(action, true).await
    }

    pub async fn advance_to_action_retain(&mut self, action: A) -> Result<()> {
        self.advance_to_action_clearable(action, false).await
    }

    pub fn add_focus_to_action(&mut self, action: A) {
        self.focus_actions.push(action);
    }

    pub fn get_focused_actions(&self) -> &[A] {
        &self.focus_actions
    }

    pub fn clear_focus(&mut self) {
        self.focus_actions.clear();
    }

    pub async fn apply_noise_at_root(&mut self) {
        let root_node_index = self.get_or_create_root_node().await;

        if let Some(dirichlet) = &self.options.dirichlet {
            let mut root_node = self.arena.get_mut();

            Self::apply_dirichlet_noise_to_node(root_node.node_mut(root_node_index), dirichlet);
        }
    }

    pub fn get_root_node(&self) -> Result<impl Deref<Target = MCTSNode<A, P, PV>> + '_> {
        let root_index = self.root.ok_or_else(|| anyhow!("No root node found!"))?;
        let arena_ref = self.arena.get();
        let node = Ref::map(arena_ref, |arena_ref| arena_ref.node(root_index));
        Ok(node)
    }

    pub fn get_node_of_edge(
        &self,
        edge: &MCTSEdge<A, PV>,
    ) -> Option<impl Deref<Target = MCTSNode<A, P, PV>> + '_> {
        edge.node_index()
            .map(|index| Ref::map(self.arena.get(), |n| n.node(index)))
    }

    pub fn get_focus_node_details(&mut self) -> Result<Option<NodeDetails<A>>> {
        self.get_focus_node_index()?
            .map(|node_index| {
                let is_root = self.focus_actions.is_empty();
                let game_state = self.get_focus_node_game_state();
                self.get_node_details(node_index, &game_state, is_root)
            })
            .transpose()
    }

    pub fn get_principal_variation(
        &mut self,
        action: Option<&A>,
        depth: usize,
    ) -> Result<Vec<(A, PUCT)>> {
        self.get_focus_node_index()?
            .map(|mut node_index| {
                let mut game_state = self.get_focus_node_game_state();
                let mut nodes = vec![];

                while nodes.len() < depth {
                    let is_root = nodes.is_empty();
                    let mut children = self
                        .get_node_details(node_index, &game_state, is_root)?
                        .children;

                    if children.is_empty() {
                        break;
                    }

                    let child_idx = if is_root && action.is_some() {
                        children
                            .iter()
                            .position(|(a, _)| a == action.unwrap())
                            .context("Action not found")?
                    } else {
                        0
                    };

                    let (action, puct) = children.swap_remove(child_idx);
                    nodes.push((action.clone(), puct));

                    if let Some(child_index) = self
                        .arena
                        .get_mut()
                        .node_mut(node_index)
                        .get_child_of_action(&action)
                        .and_then(|child| child.node_index())
                    {
                        game_state = self.game_engine.take_action(&game_state, &action);
                        node_index = child_index;
                        continue;
                    }

                    break;
                }

                Ok(nodes)
            })
            .context("Focused action was not found")?
    }

    pub fn num_focus_node_visits(&mut self) -> usize {
        self.get_focus_node_index()
            .ok()
            .flatten()
            .map(|node_index| self.arena.get_mut().node(node_index).get_node_visits())
            .unwrap_or(0)
    }

    fn get_node_details(
        &mut self,
        node_index: Index,
        game_state: &S,
        is_root: bool,
    ) -> Result<NodeDetails<A>> {
        let arena_ref_mut = &mut self.arena.get_mut();
        let metrics = Self::get_PUCT_for_nodes(
            node_index,
            game_state,
            arena_ref_mut,
            is_root,
            &self.options,
        );

        let node = arena_ref_mut.node_mut(node_index);

        let mut children: Vec<_> = node
            .iter_all_edges()
            .zip(metrics)
            .map(|(n, m)| (n.action().clone(), m))
            .collect();

        children.sort_by(|(_, x_puct), (_, y_puct)| y_puct.cmp(x_puct));

        Ok(NodeDetails {
            visits: node.get_node_visits(),
            children,
        })
    }

    fn _select_action(&mut self, use_temp: bool) -> Result<A> {
        if let Some(node_index) = &self.get_focus_node_index()? {
            let game_state = self.get_focus_node_game_state();
            let temp = self.temp.temp(&game_state);

            let child_node_details = self
                .get_node_details(*node_index, &game_state, self.focus_actions.is_empty())?
                .children;

            if child_node_details.is_empty() {
                return Err(anyhow!("Node has no children. This node should have been designated as a terminal node. {:?}", game_state));
            }

            let best_action = if temp == 0.0 || !use_temp {
                let (best_action, _) = child_node_details
                    .first()
                    .ok_or_else(|| anyhow!("No available actions"))?;
                best_action
            } else {
                let candidates: Vec<_> = child_node_details
                    .iter()
                    .map(|(a, puct)| (a, puct.Nsa))
                    .collect();
                let chosen_index = Self::select_action_using_temperature(
                    &candidates,
                    temp,
                    self.options.temperature_visit_offset,
                )?;
                candidates[chosen_index].0
            };

            return Ok(best_action.clone());
        }

        Err(anyhow!(
            "Root or focused node does not exist. Run search first."
        ))
    }

    #[allow(clippy::await_holding_refcell_ref)]
    async fn advance_to_action_clearable(&mut self, action: A, clear: bool) -> Result<()> {
        self.clear_focus();

        let root_index = self.get_or_create_root_node().await;

        let game_engine = &self.game_engine;

        let mut arena_mut = self.arena.get_mut();
        let mut root_node = arena_mut.remove(root_index);
        let split_nodes = Self::split_node_children_by_action(&mut root_node, &action);

        if let Err(err) = split_nodes {
            // If there is an error, replace the root node back to it's original value.
            let index = arena_mut.insert(root_node);
            self.root = Some(index);
            return Err(err);
        }

        self.game_state = game_engine.take_action(&self.game_state, &action);

        let (chosen_node_index, other_nodes_indexes) =
            split_nodes.expect("Expected node to exist.");

        for node_index in other_nodes_indexes.into_iter() {
            Self::remove_nodes_from_arena(node_index, &mut arena_mut);
        }

        drop(arena_mut);

        let chosen_node_index = if let Some(node_index) = chosen_node_index {
            if clear {
                Self::clear_node(node_index, &mut self.arena.get_mut());
            }

            node_index
        } else {
            let node = Self::analyse_and_create_node(&self.game_state, self.analyzer).await;
            self.arena.get_mut().insert(node)
        };

        self.root.replace(chosen_node_index);

        Ok(())
    }

    fn remove_nodes_from_arena(node_index: Index, arena: &mut NodeArenaInner<MCTSNode<A, P, PV>>) {
        let node = arena.remove(node_index);

        for child_node_index in node.iter_visited_edges().filter_map(|n| n.node_index()) {
            Self::remove_nodes_from_arena(child_node_index, arena);
        }
    }

    fn clear_node(node_index: Index, arena: &mut NodeArenaInner<MCTSNode<A, P, PV>>) {
        let node = arena.node_mut(node_index);
        node.set_visits(1);

        for child in node.iter_visited_edges_mut() {
            child.clear();
        }

        let child_indexes: Vec<_> = node
            .iter_visited_edges()
            .filter_map(|c| c.node_index())
            .collect();

        for child_index in child_indexes {
            Self::clear_node(child_index, arena);
        }
    }

    fn split_node_children_by_action(
        current_root: &mut MCTSNode<A, P, PV>,
        action: &A,
    ) -> Result<(Option<Index>, Vec<Index>)> {
        let matching_action = current_root
            .get_child_of_action(action)
            .ok_or_else(|| anyhow!("No matching Action"))?
            .node_index();

        let other_actions: Vec<_> = current_root
            .iter_visited_edges()
            .filter(|n| n.action() != action)
            .filter_map(|n| n.node_index())
            .collect();

        Ok((matching_action, other_actions))
    }

    fn select_action_using_temperature(
        action_visits: &[(&A, usize)],
        temp: f32,
        temperature_visit_offset: f32,
    ) -> Result<usize> {
        let normalized_visits = action_visits.iter().map(|(_, visits)| {
            (*visits as f32 + temperature_visit_offset)
                .max(0.0)
                .powf(1.0 / temp)
        });

        let weighted_index = WeightedIndex::new(normalized_visits);

        let chosen_idx = match weighted_index {
            Err(_) => {
                warn!(
                    "Invalid puct scores. Most likely all are 0. Move will be randomly selected."
                );
                warn!("{:?}", action_visits);
                thread_rng().gen_range(0..action_visits.len())
            }
            Ok(weighted_index) => weighted_index.sample(&mut thread_rng()),
        };

        Ok(chosen_idx)
    }

    async fn get_or_create_root_node(&mut self) -> Index {
        let root = &mut self.root;

        if let Some(root_node_index) = root.as_ref() {
            return *root_node_index;
        }

        let root_node = Self::analyse_and_create_node(&self.game_state, self.analyzer).await;

        let root_node_index = self.arena.get_mut().insert(root_node);

        root.replace(root_node_index);

        root_node_index
    }

    async fn analyse_and_create_node(game_state: &S, analyzer: &M) -> MCTSNode<A, P, PV> {
        let analysis = analyzer.get_state_analysis(game_state).await;
        let (policy_scores, predictions) = analysis.into_inner();
        MCTSNode::new(policy_scores, predictions)
    }

    fn apply_dirichlet_noise_to_node(node: &mut MCTSNode<A, P, PV>, dirichlet: &DirichletOptions) {
        let policy_scores: Vec<f32> = node
            .iter_all_edges()
            .map(|child_node| child_node.policy_score())
            .collect();

        let noisy_policy_scores = Self::generate_noise(policy_scores, dirichlet);

        for (child, noisy_policy_score) in
            node.iter_all_edges().zip(noisy_policy_scores.into_iter())
        {
            child.set_policy_score(noisy_policy_score);
        }
    }

    fn get_focus_node_index(&mut self) -> Result<Option<Index>> {
        let mut arena_ref = self.arena.get_mut();

        let mut node_index = *self
            .root
            .as_ref()
            .ok_or_else(|| anyhow!("No root node found!"))?;

        for action in &self.focus_actions {
            match arena_ref
                .node_mut(node_index)
                .get_child_of_action(action)
                .and_then(|child| child.node_index())
            {
                Some(child_index) => node_index = child_index,
                None => return Ok(None),
            };
        }

        Ok(Some(node_index))
    }

    fn get_focus_node_game_state(&self) -> S {
        let mut game_state = Cow::Borrowed(&self.game_state);

        for action in &self.focus_actions {
            game_state = Cow::Owned(self.game_engine.take_action(&*game_state, action))
        }

        game_state.into_owned()
    }

    fn generate_noise(policy_scores: Vec<f32>, dirichlet: &DirichletOptions) -> Vec<f32> {
        let num_actions = policy_scores.len();

        // Do not apply noise if there is only one action.
        if num_actions < 2 {
            return policy_scores;
        }

        let e = dirichlet.epsilon;
        let alpha = 8.0 / num_actions as f32;
        let dirichlet_noise = Dirichlet::new_with_size(alpha, num_actions)
            .expect("Error creating dirichlet distribution")
            .sample(&mut thread_rng());

        dirichlet_noise
            .into_iter()
            .zip(policy_scores)
            .map(|(noise, policy_score)| (1.0 - e) * policy_score + e * noise)
            .collect()
    }
}

impl<'a, S, A, E, M, B, Sel, P, T, PV> MCTS<'a, S, A, E, M, B, Sel, P, T, PV>
where
    S: GameState,
    A: Clone + Eq + Debug,
    E: 'a + GameEngine<State = S, Action = A, Terminal = P>,
    M: 'a + GameAnalyzer<State = S, Action = A, Predictions = P>,
    B: 'a + BackpropagationStrategy<State = S, Action = A, Predictions = P, PredicationValues = PV>,
    Sel: 'a + SelectionStrategy<State = S, Action = A, Predictions = P, PredicationValues = PV>,
    T: Temperature<State = S>,
    PV: Default,
    B::NodeInfo: Clone,
{
    pub async fn search_time(&mut self, duration: Duration) -> Result<usize> {
        self.search_time_max_visits(duration, usize::max_value())
            .await
    }

    pub async fn search_time_max_visits(
        &mut self,
        duration: Duration,
        max_visits: usize,
    ) -> Result<usize> {
        let alive = Arc::new(AtomicBool::new(true));

        let alive_clone = alive.clone();
        thread::spawn(move || {
            thread::sleep(duration);
            alive_clone.store(false, Ordering::SeqCst);
        });

        self.search(|visits| alive.load(Ordering::SeqCst) && visits < max_visits)
            .await
    }

    pub async fn search_visits(&mut self, visits: usize) -> Result<usize> {
        self.search(|node_visits| node_visits < visits).await
    }

    pub async fn play(&mut self, alive: &mut bool) -> Result<usize> {
        self.search(|_| *alive).await
    }

    pub async fn search<Fn>(&mut self, alive: Fn) -> Result<usize>
    where
        Fn: FnMut(usize) -> bool,
    {
        let root_node_index = self.get_or_create_root_node().await;

        let mut visits = self.num_focus_node_visits();

        let game_engine = &self.game_engine;
        let arena_ref = &self.arena;
        let focus_actions = &self.focus_actions;
        let game_state = &self.game_state;
        let mut max_depth: usize = 0;
        let mut alive_flag = true;
        let mut alive = alive;
        let mut searches = FuturesUnordered::new();

        let analyzer = &mut self.analyzer;
        let selection_strategy = &self.selection_strategy;
        let backpropagation_strategy = &self.backpropagation_strategy;

        let mut traverse = |searches: &mut FuturesUnordered<_>| {
            if alive_flag && alive(visits) {
                searches.push(Self::traverse_tree_and_expand(
                    root_node_index,
                    arena_ref,
                    game_state,
                    focus_actions,
                    game_engine,
                    analyzer,
                    selection_strategy,
                    backpropagation_strategy,
                ));
                visits += 1;
            } else {
                alive_flag = false;
            }
        };

        for _ in 0..self.options.parallelism {
            traverse(&mut searches);
        }

        while let Some(search_depth) = searches.next().await {
            traverse(&mut searches);

            max_depth = max_depth.max(search_depth?);
        }

        Ok(max_depth)
    }

    #[allow(clippy::await_holding_refcell_ref)]
    #[allow(clippy::too_many_arguments)]
    async fn traverse_tree_and_expand(
        root_index: Index,
        arena: &NodeArena<MCTSNode<A, P, PV>>,
        game_state: &S,
        focus_actions: &[A],
        game_engine: &E,
        analyzer: &M,
        selection_strategy: &Sel,
        backpropagation_strategy: &B,
    ) -> Result<usize> {
        let mut depth = 0;
        let mut nodes_to_propagate_to_stack: Vec<NodeUpdateInfo<B::NodeInfo>> = vec![];
        let mut latest_index = root_index;
        let mut game_state = Cow::Borrowed(game_state);

        loop {
            depth += 1;
            let mut arena_mut = arena.get_mut();
            let node = arena_mut.node_mut(latest_index);

            if node.is_terminal() {
                node.increment_visits();
                Self::backpropagate(
                    &nodes_to_propagate_to_stack,
                    latest_index,
                    backpropagation_strategy,
                    &mut *arena_mut,
                );
                break;
            }

            let is_root = depth == 1;

            let selected_edge_index = if let Some(focus_action) = focus_actions.get(depth - 1) {
                node.get_position_of_visited_action(focus_action)
                    .ok_or_else(|| anyhow!("Focused action was not found"))?
            } else {
                selection_strategy.select_path(node, &game_state, is_root)?
            };

            let node_info = backpropagation_strategy.node_info(&game_state);
            nodes_to_propagate_to_stack.push(NodeUpdateInfo {
                node_index: latest_index,
                node_info,
                selected_edge_index: selected_edge_index,
            });

            node.increment_visits();
            let selected_edge = node.get_edge_by_index_mut(selected_edge_index);

            game_state = Cow::Owned(game_engine.take_action(&game_state, selected_edge.action()));

            let prev_visits = selected_edge.visits();
            selected_edge.increment_visits();

            if let Some(selected_child_node_index) = selected_edge.node_index() {
                // If the node exists but visits was 0, then this node was cleared but the analysis was saved. Treat it as such by keeping the values.
                if prev_visits == 0 {
                    Self::backpropagate(
                        &nodes_to_propagate_to_stack,
                        latest_index,
                        backpropagation_strategy,
                        &mut *arena_mut,
                    );
                    break;
                }

                // Continue with the next iteration of the loop since we found an already expanded child node.
                latest_index = selected_child_node_index;
                continue;
            }

            // If the node is yet to be expanded and is not already expanded, then start expanding it.
            if selected_edge.is_unexpanded() {
                selected_edge.mark_as_expanding();
                drop(arena_mut);

                let expanded_node = Self::analyse_and_create_node(&game_state, analyzer).await;

                let mut arena_mut = arena.get_mut();
                let expanded_node_index = arena_mut.insert(expanded_node);

                let selected_child_node = arena_mut.child(latest_index, selected_edge_index);

                selected_child_node.set_expanded(expanded_node_index);

                Self::backpropagate(
                    &nodes_to_propagate_to_stack,
                    latest_index,
                    backpropagation_strategy,
                    &mut *arena_mut,
                );

                break;
            }

            // The node must be expanding in another async operation. Wait for that to finish and continue the loop as if it was already expanded.
            let waiter = selected_edge.get_waiter();
            drop(arena_mut);

            waiter.wait().await;

            latest_index = arena
                .get_mut()
                .child(latest_index, selected_edge_index)
                .node_index()
                .expect("Node should have expanded");
        }

        Ok(depth)
    }

    fn backpropagate(
        node_update_info: &[NodeUpdateInfo<B::NodeInfo>],
        evaluated_node_index: Index,
        backpropagation_strategy: &B,
        arena: &mut NodeArenaInner<MCTSNode<A, P, PV>>,
    ) {
        // let node_iter = NodeIterator {
        //     node_indexes: node_update_info
        //         .iter()
        //         .map(|info| info.node_index)
        //         .collect(),
        //     arena,
        // };

        let node_and_info_iter = node_update_info
            .iter()
            .map(|info| (info.node_info.clone(), arena.node_mut(info.node_index)));

        let evaluated_node = arena.node_mut(evaluated_node_index);

        backpropagation_strategy.backpropagate(node_and_info_iter, evaluated_node);
    }
}

// struct NodeIterator<'a, A, P, PV> {
//     node_indexes: Vec<Index>,
//     arena: &'a mut NodeArenaInner<MCTSNode<A, P, PV>>,
// }

// impl<'a, 'b, A, P, PV> Iterator for NodeIterator<'b, A, P, PV> {
//     type Item = &'a mut MCTSNode<A, P, PV>;

//     fn next(&mut self) -> Option<Self::Item> {
//         self.node_indexes
//             .pop()
//             .map(|node_index| self.arena.node_mut(node_index))
//     }
// }

impl<'a, S, A, E, M, B, Sel, P, T, PV> MCTS<'a, S, A, E, M, B, Sel, P, T, PV>
where
    A: Clone,
    P: Clone,
    PV: Clone + Default,
{
    pub fn get_root_node_metrics(&mut self) -> Result<NodeMetrics<A, P, PV>> {
        let root_index = self.root.ok_or_else(|| anyhow!("No root node found!"))?;
        let mut arena_ref = self.arena.get_mut();
        let root = arena_ref.node(root_index);

        Ok(root.into())
    }
}

impl<A, P, PV> From<&MCTSNode<A, P, PV>> for NodeMetrics<A, P, PV>
where
    A: Clone,
    P: Clone,
    PV: Clone + Default,
{
    fn from(val: &MCTSNode<A, P, PV>) -> Self {
        NodeMetrics {
            visits: val.visits(),
            predictions: val.predictions().clone(),
            children: val.iter_all_edges().map(|e| e.deref().into()).collect_vec(),
        }
    }
}

impl<A, PV> From<&MCTSEdge<A, PV>> for EdgeMetrics<A, PV>
where
    A: Clone,
    PV: Clone,
{
    fn from(edge: &MCTSEdge<A, PV>) -> Self {
        EdgeMetrics::new(
            edge.action().clone(),
            edge.visits(),
            edge.propagated_values().clone(),
        )
    }
}

struct NodeUpdateInfo<I> {
    node_index: Index,
    selected_edge_index: usize,
    node_info: I,
}

struct NodeArena<T>(RefCell<NodeArenaInner<T>>);

impl<T> NodeArena<T> {
    fn with_capacity(n: usize) -> Self {
        Self(RefCell::new(NodeArenaInner::with_capacity(n)))
    }

    #[inline]
    fn get(&self) -> Ref<'_, NodeArenaInner<T>> {
        self.0.borrow()
    }

    #[inline]
    fn get_mut(&self) -> RefMut<'_, NodeArenaInner<T>> {
        self.0.borrow_mut()
    }
}

struct NodeArenaInner<T>(Arena<T>);

impl<T> NodeArenaInner<T> {
    fn with_capacity(n: usize) -> Self {
        Self(Arena::with_capacity(n))
    }
}

impl<A, P, PV> NodeArenaInner<MCTSNode<A, P, PV>> {
    #[inline]
    fn node(&self, index: Index) -> &MCTSNode<A, P, PV> {
        &self.0[index]
    }

    #[inline]
    fn node_mut(&mut self, index: Index) -> &mut MCTSNode<A, P, PV> {
        &mut self.0[index]
    }

    #[inline]
    fn child(&mut self, index: Index, edge_index: usize) -> &mut MCTSEdge<A, PV> {
        self.0[index].get_edge_by_index_mut(edge_index)
    }

    #[inline]
    fn insert(&mut self, node: MCTSNode<A, P, PV>) -> Index {
        self.0.insert(node)
    }

    #[inline]
    fn remove(&mut self, index: Index) -> MCTSNode<A, P, PV> {
        self.0
            .remove(index)
            .expect("Expected node to exist in the arena")
    }
}
