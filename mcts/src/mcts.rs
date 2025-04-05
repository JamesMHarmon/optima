use crate::{SelectedNode, TempAndOffset};

use super::Temperature;
use super::{BackpropagationStrategy, EdgeDetails, NodeLendingIterator, SelectionStrategy};
use super::{DirichletOptions, MCTSEdge, MCTSNode, NodeDetails};
use anyhow::{anyhow, Context, Result};
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

pub struct MCTS<'a, S, A, E, M, B, Sel, P, PV> {
    game_engine: &'a E,
    analyzer: &'a M,
    backpropagation_strategy: &'a B,
    selection_strategy: &'a Sel,
    game_state: S,
    root: Option<Index>,
    arena: NodeArena<MCTSNode<A, P, PV>>,
    focus_actions: Vec<A>,
    parallelism: usize,
}

#[allow(non_snake_case)]
impl<'a, S, A, E, M, B, Sel, P, PV> MCTS<'a, S, A, E, M, B, Sel, P, PV>
where
    S: GameState,
    A: Clone + Eq + Debug,
    E: 'a + GameEngine<State = S, Action = A, Terminal = P>,
    M: 'a + GameAnalyzer<State = S, Action = A, Predictions = P>,
    B: 'a,
    Sel: 'a,
    PV: Default,
{
    pub fn new(
        game_state: S,
        game_engine: &'a E,
        analyzer: &'a M,
        backpropagation_strategy: &'a B,
        selection_strategy: &'a Sel,
        parallelism: usize,
    ) -> Self {
        MCTS {
            game_engine,
            analyzer,
            backpropagation_strategy,
            selection_strategy,
            game_state,
            root: None,
            arena: NodeArena::with_capacity(800 * 2),
            focus_actions: vec![],
            parallelism,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn with_capacity(
        game_state: S,
        game_engine: &'a E,
        analyzer: &'a M,
        backpropagation_strategy: &'a B,
        selection_strategy: &'a Sel,
        capacity: usize,
        parallelism: usize,
    ) -> Self {
        MCTS {
            game_engine,
            analyzer,
            backpropagation_strategy,
            selection_strategy,
            game_state,
            root: None,
            arena: NodeArena::with_capacity(capacity * 2),
            focus_actions: vec![],
            parallelism,
        }
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

    pub async fn apply_noise_at_root(&mut self, dirichlet: Option<&DirichletOptions>) {
        let root_node_index = self.get_or_create_root_node().await;

        if let Some(dirichlet) = dirichlet {
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

    pub fn num_focus_node_visits(&mut self) -> usize {
        self.get_focus_node_index()
            .ok()
            .flatten()
            .map(|node_index| self.arena.get_mut().node(node_index).visits())
            .unwrap_or(0)
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

        for edge in node.iter_visited_edges_mut() {
            edge.clear();
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
        edge_details: &[EdgeDetails<A, PV>],
        tempAndOffset: TempAndOffset,
    ) -> Result<A> {
        let normalized_visits = edge_details.iter().map(|edge_details| {
            (edge_details.Nsa as f32 + tempAndOffset.temperature_visit_offset)
                .max(0.0)
                .powf(1.0 / tempAndOffset.temperature)
        });

        let weighted_index = WeightedIndex::new(normalized_visits);

        let chosen_idx = match weighted_index {
            Err(_) => {
                warn!(
                    "Invalid puct scores. Most likely all are 0. Move will be randomly selected."
                );
                thread_rng().gen_range(0..edge_details.len())
            }
            Ok(weighted_index) => weighted_index.sample(&mut thread_rng()),
        };

        Ok(edge_details[chosen_idx].action.clone())
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

impl<'a, S, A, E, M, B, Sel, P, PV> MCTS<'a, S, A, E, M, B, Sel, P, PV>
where
    S: GameState,
    A: Clone + Eq + Debug,
    E: 'a + GameEngine<State = S, Action = A, Terminal = P>,
    M: 'a + GameAnalyzer<State = S, Action = A, Predictions = P>,
    Sel: 'a + SelectionStrategy<State = S, Action = A, Predictions = P, PropagatedValues = PV>,
    PV: Default + Ord,
{
    pub fn select_action<T>(&mut self, temp: &T) -> Result<A>
    where
        T: Temperature<State = S>,
    {
        if let Some(node_index) = &self.get_focus_node_index()? {
            let game_state = self.get_focus_node_game_state();

            let child_node_details = self
                .node_details(*node_index, &game_state, self.focus_actions.is_empty())?
                .children;

            if child_node_details.is_empty() {
                return Err(anyhow!("Node has no children. This node should have been designated as a terminal node. {:?}", game_state));
            }

            let temp_and_offset = temp.temp(&game_state);

            let best_action = if temp_and_offset.temperature > 0.0 {
                Self::select_action_using_temperature(&child_node_details, temp_and_offset)?
            } else {
                let best_action = child_node_details
                    .into_iter()
                    .next()
                    .ok_or_else(|| anyhow!("No available actions"))?;
                best_action.action
            };

            return Ok(best_action);
        }

        Err(anyhow!(
            "Root or focused node does not exist. Run search first."
        ))
    }

    pub fn get_focus_node_details(&mut self) -> Result<Option<NodeDetails<A, PV>>> {
        self.get_focus_node_index()?
            .map(|node_index| {
                let is_root = self.focus_actions.is_empty();
                let game_state = self.get_focus_node_game_state();
                self.node_details(node_index, &game_state, is_root)
            })
            .transpose()
    }

    pub fn get_principal_variation(
        &mut self,
        action: Option<&A>,
        depth: usize,
    ) -> Result<Vec<EdgeDetails<A, PV>>> {
        self.get_focus_node_index()?
            .map(|mut node_index| {
                let mut game_state = self.get_focus_node_game_state();
                let mut nodes = vec![];

                while nodes.len() < depth {
                    let is_root = nodes.is_empty();
                    let mut children = self
                        .node_details(node_index, &game_state, is_root)?
                        .children;

                    if children.is_empty() {
                        break;
                    }

                    let child_idx = if is_root && action.is_some() {
                        children
                            .iter()
                            .position(|details| &details.action == action.unwrap())
                            .context("Action not found")?
                    } else {
                        0
                    };

                    let details = children.swap_remove(child_idx);
                    let action = details.action.clone();
                    nodes.push(details);

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

    fn node_details(
        &mut self,
        node_index: Index,
        game_state: &S,
        is_root: bool,
    ) -> Result<NodeDetails<A, PV>> {
        let arena_ref = &mut self.arena.get_mut();
        let node = arena_ref.node_mut(node_index);
        let mut children = self
            .selection_strategy
            .node_details(node, game_state, is_root);

        children.sort_by(|x_details, y_details| y_details.cmp(x_details));

        Ok(NodeDetails {
            visits: node.visits(),
            children,
        })
    }
}

impl<'a, S, A, E, M, B, Sel, P, PV> MCTS<'a, S, A, E, M, B, Sel, P, PV>
where
    S: GameState,
    A: Clone + Eq + Debug,
    E: 'a + GameEngine<State = S, Action = A, Terminal = P>,
    M: 'a + GameAnalyzer<State = S, Action = A, Predictions = P>,
    B: 'a + BackpropagationStrategy<State = S, Action = A, Predictions = P, PropagatedValues = PV>,
    Sel: 'a + SelectionStrategy<State = S, Action = A, Predictions = P, PropagatedValues = PV>,
    P: Clone,
    PV: Default,
{
    pub async fn search_time(&mut self, duration: Duration) -> Result<usize> {
        self.search_time_max_visits(duration, usize::MAX).await
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

        for _ in 0..self.parallelism {
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
        let mut visited_nodes_stack: Vec<NodeUpdateInfo<B::NodeInfo>> = vec![];
        let mut latest_index = root_index;
        let mut game_state = Cow::Borrowed(game_state);

        loop {
            depth += 1;
            let mut arena_mut = arena.get_mut();
            let node = arena_mut.node_mut(latest_index);

            if node.is_terminal() {
                node.increment_visits();
                Self::backpropagate(
                    &node.predictions().clone(),
                    &visited_nodes_stack,
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

            node.increment_visits();
            let selected_edge = node.get_edge_by_index_mut(selected_edge_index);
            let edge_visits = selected_edge.visits();

            visited_nodes_stack.push(NodeUpdateInfo {
                node_index: latest_index,
                selected_edge_index,
                node_info,
            });

            game_state = Cow::Owned(game_engine.take_action(&game_state, selected_edge.action()));
            selected_edge.increment_visits();

            if let Some(selected_child_node_index) = selected_edge.node_index() {
                // If the node exists but visits was 0, then this node was cleared but the analysis was saved. Treat it as such by keeping the values.
                if edge_visits == 0 {
                    let predictions = arena_mut
                        .node(selected_child_node_index)
                        .predictions()
                        .clone();
                    Self::backpropagate(
                        &predictions,
                        &visited_nodes_stack,
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
                let predictions = expanded_node.predictions().clone();

                let mut arena_mut = arena.get_mut();
                let expanded_node_index = arena_mut.insert(expanded_node);

                let selected_child_node = arena_mut.child(latest_index, selected_edge_index);

                selected_child_node.set_expanded(expanded_node_index);

                Self::backpropagate(
                    &predictions,
                    &visited_nodes_stack,
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
        predictions: &P,
        visited_node_info: &[NodeUpdateInfo<B::NodeInfo>],
        backpropagation_strategy: &B,
        arena: &mut NodeArenaInner<MCTSNode<A, P, PV>>,
    ) {
        let node_iter = NodeIterator::new(visited_node_info, arena);
        backpropagation_strategy.backpropagate(node_iter, predictions);
    }
}

pub struct NodeIterator<'arena, 'node, I, A, P, PV> {
    visited_node_info: &'node [NodeUpdateInfo<I>],
    arena: &'arena mut NodeArenaInner<MCTSNode<A, P, PV>>,
    iter_index: usize,
}

impl<'arena, 'node, I, A, P, PV> NodeIterator<'arena, 'node, I, A, P, PV> {
    fn new(
        visited_node_info: &'node [NodeUpdateInfo<I>],
        arena: &'arena mut NodeArenaInner<MCTSNode<A, P, PV>>,
    ) -> Self {
        Self {
            visited_node_info,
            arena,
            iter_index: 0,
        }
    }
}

impl<'arena, 'node, I, A, P, PV> NodeLendingIterator<'node, I, A, P, PV>
    for NodeIterator<'arena, 'node, I, A, P, PV>
{
    fn next(&mut self) -> Option<SelectedNode<I, A, P, PV>> {
        if self.iter_index >= self.visited_node_info.len() {
            return None;
        }

        let node = &self.visited_node_info[self.iter_index];
        let selected_node = SelectedNode {
            node: self.arena.node_mut(node.node_index),
            selected_edge_index: node.selected_edge_index,
            node_info: &node.node_info,
        };

        self.iter_index += 1;

        Some(selected_node)
    }
}

impl<'a, S, A, E, M, B, Sel, P, PV> MCTS<'a, S, A, E, M, B, Sel, P, PV>
where
    A: Clone,
    P: Clone,
    PV: Clone + Default,
{
    pub fn get_root_node_metrics(&mut self) -> Result<NodeMetrics<A, P, PV>> {
        let root_index = self.root.ok_or_else(|| anyhow!("No root node found!"))?;
        let mut arena_ref = self.arena.get_mut();
        let root = arena_ref.node_mut(root_index);

        Ok(root.into())
    }
}

impl<A, P, PV> From<&mut MCTSNode<A, P, PV>> for NodeMetrics<A, P, PV>
where
    A: Clone,
    P: Clone,
    PV: Clone + Default,
{
    fn from(node: &mut MCTSNode<A, P, PV>) -> Self {
        NodeMetrics {
            visits: node.visits(),
            predictions: node.predictions().clone(),
            children: node
                .iter_all_edges()
                .map(|e| e.deref().into())
                .collect_vec(),
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
