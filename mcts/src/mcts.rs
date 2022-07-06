use anyhow::{anyhow, Result};
use common::div_or_zero;
use futures::stream::{FuturesUnordered, StreamExt};
use futures_intrusive::sync::LocalManualResetEvent;
use generational_arena::{Arena, Index};
use itertools::Itertools;
use log::warn;
use model::NodeChildMetrics;
use rand::distributions::WeightedIndex;
use rand::prelude::Distribution;
use rand::{thread_rng, Rng};
use rand_distr::Dirichlet;
use std::borrow::Cow;
use std::cell::Ref;
use std::cell::RefCell;
use std::cell::RefMut;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::Deref;
use std::rc::Rc;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::thread;
use std::time::Duration;

use super::node_details::{NodeDetails, PUCT};
use engine::engine::GameEngine;
use engine::game_state::GameState;
use engine::value::Value;
use model::analytics::{ActionWithPolicy, GameAnalyzer, GameStateAnalysis};
use model::node_metrics::NodeMetrics;

pub struct DirichletOptions {
    pub epsilon: f32,
}

pub struct MCTSOptions<S, C, T>
where
    S: GameState,
    C: Fn(&S, usize, bool) -> f32,
    T: Fn(&S) -> f32,
{
    dirichlet: Option<DirichletOptions>,
    fpu: f32,
    fpu_root: f32,
    logit_q: bool,
    cpuct: C,
    temperature: T,
    temperature_visit_offset: f32,
    moves_left_threshold: f32,
    moves_left_scale: f32,
    moves_left_factor: f32,
    parallelism: usize,
    _phantom_state: PhantomData<*const S>,
}

#[allow(clippy::too_many_arguments)]
impl<S, C, T> MCTSOptions<S, C, T>
where
    S: GameState,
    C: Fn(&S, usize, bool) -> f32,
    T: Fn(&S) -> f32,
{
    pub fn new(
        dirichlet: Option<DirichletOptions>,
        fpu: f32,
        fpu_root: f32,
        logit_q: bool,
        cpuct: C,
        temperature: T,
        temperature_visit_offset: f32,
        moves_left_threshold: f32,
        moves_left_scale: f32,
        moves_left_factor: f32,
        parallelism: usize,
    ) -> Self {
        MCTSOptions {
            dirichlet,
            fpu,
            fpu_root,
            logit_q,
            cpuct,
            temperature,
            temperature_visit_offset,
            moves_left_threshold,
            moves_left_scale,
            moves_left_factor,
            parallelism,
            _phantom_state: PhantomData,
        }
    }
}

pub struct MCTS<'a, S, A, E, M, C, T, V>
where
    S: GameState,
    A: Clone + Eq + Debug,
    V: Value,
    E: GameEngine,
    M: GameAnalyzer,
    C: Fn(&S, usize, bool) -> f32,
    T: Fn(&S) -> f32,
{
    options: MCTSOptions<S, C, T>,
    game_engine: &'a E,
    analyzer: &'a M,
    game_state: S,
    root: Option<Index>,
    arena: NodeArena<MCTSNode<A, V>>,
    focus_actions: Vec<A>,
}

#[allow(non_snake_case)]
impl<'a, S, A, E, M, C, T, V> MCTS<'a, S, A, E, M, C, T, V>
where
    S: GameState,
    A: Clone + Eq + Debug,
    V: Value,
    E: 'a + GameEngine<State = S, Action = A, Value = V>,
    M: 'a + GameAnalyzer<State = S, Action = A, Value = V>,
    C: Fn(&S, usize, bool) -> f32,
    T: Fn(&S) -> f32,
{
    pub fn new(
        game_state: S,
        game_engine: &'a E,
        analyzer: &'a M,
        options: MCTSOptions<S, C, T>,
    ) -> Self {
        MCTS {
            options,
            game_engine,
            analyzer,
            game_state,
            root: None,
            arena: NodeArena::with_capacity(800 * 2),
            focus_actions: vec![],
        }
    }

    pub fn with_capacity(
        game_state: S,
        game_engine: &'a E,
        analyzer: &'a M,
        options: MCTSOptions<S, C, T>,
        capacity: usize,
    ) -> Self {
        MCTS {
            options,
            game_engine,
            analyzer,
            game_state,
            root: None,
            arena: NodeArena::with_capacity(capacity * 2),
            focus_actions: vec![],
        }
    }

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

    pub fn get_root_node_metrics(&mut self) -> Result<NodeMetrics<A, V>> {
        let root_index = self.root.ok_or_else(|| anyhow!("No root node found!"))?;
        let arena_ref = self.arena.get_mut();
        let root = arena_ref.node(root_index);

        Ok(root.into())
    }

    pub fn get_root_node(&self) -> Result<impl Deref<Target = MCTSNode<A, V>> + '_> {
        let root_index = self.root.ok_or_else(|| anyhow!("No root node found!"))?;
        let arena_ref = self.arena.get();
        let node = Ref::map(arena_ref, |arena_ref| arena_ref.node(root_index));
        Ok(node)
    }

    pub fn get_node_of_edge(
        &self,
        edge: &MCTSEdge<A>,
    ) -> Option<impl Deref<Target = MCTSNode<A, V>> + '_> {
        edge.node_index()
            .map(|index| Ref::map(self.arena.get(), |n| n.node(index)))
    }

    pub fn get_focus_node_details(&self) -> Result<Option<NodeDetails<A>>> {
        self.get_focus_node_index()?
            .map(|node_index| {
                let is_root = self.focus_actions.is_empty();
                let game_state = self.get_focus_node_game_state();
                self.get_node_details(node_index, &game_state, is_root)
            })
            .transpose()
    }

    pub fn get_principal_variation(&self) -> Result<Vec<(A, PUCT)>> {
        self.get_focus_node_index()?
            .map(|mut node_index| {
                let mut game_state = self.get_focus_node_game_state();
                let mut nodes = vec![];
                let arena_ref = self.arena.get();

                loop {
                    let is_root = nodes.is_empty();
                    let mut children = self
                        .get_node_details(node_index, &game_state, is_root)?
                        .children;

                    if children.is_empty() {
                        break;
                    }

                    let (action, puct) = children.swap_remove(0);
                    nodes.push((action.clone(), puct));

                    if let Some(child_index) = arena_ref
                        .node(node_index)
                        .get_child_of_action(&action)
                        .and_then(|child| child.node.get_index())
                    {
                        game_state = self.game_engine.take_action(&game_state, &action);
                        node_index = child_index;
                        continue;
                    }

                    break;
                }

                Ok(nodes)
            })
            .ok_or_else(|| anyhow!("Focused action was not found"))
            .flatten()
    }

    pub async fn search<F: FnMut(usize) -> bool>(&mut self, alive: F) -> Result<usize> {
        let root_node_index = self.get_or_create_root_node().await;

        let game_engine = &self.game_engine;
        let options = &self.options;
        let arena_ref = &self.arena;
        let focus_actions = &self.focus_actions;
        let game_state = &self.game_state;
        let mut max_depth: usize = 0;
        let mut alive_flag = true;
        let mut alive = alive;
        let mut searches = FuturesUnordered::new();
        let mut visits = self
            .get_focus_node_index()?
            .map(|node_index| arena_ref.get_mut().node(node_index).get_node_visits())
            .unwrap_or(0);
        let analyzer = &mut self.analyzer;

        let mut traverse = |searches: &mut FuturesUnordered<_>| {
            if alive_flag && alive(visits) {
                searches.push(Self::traverse_tree_and_expand(
                    root_node_index,
                    arena_ref,
                    game_state,
                    focus_actions,
                    game_engine,
                    analyzer,
                    options,
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
    async fn traverse_tree_and_expand(
        root_index: Index,
        arena: &NodeArena<MCTSNode<A, V>>,
        game_state: &S,
        focus_actions: &[A],
        game_engine: &E,
        analyzer: &M,
        options: &MCTSOptions<S, C, T>,
    ) -> Result<usize> {
        let mut depth = 0;
        let mut nodes_to_propagate_to_stack: Vec<NodeUpdateInfo> = vec![];
        let mut latest_index = root_index;
        let mut game_state = Cow::Borrowed(game_state);
        let mut move_number = game_engine.get_move_number(&game_state);

        loop {
            depth += 1;
            let mut arena_mut = arena.get_mut();
            let node = arena_mut.node_mut(latest_index);

            if node.is_terminal() {
                node.visits += 1;
                Self::update_node_values(
                    &nodes_to_propagate_to_stack,
                    latest_index,
                    move_number,
                    &mut *arena_mut,
                );
                break;
            }

            let is_root = depth == 1;

            let selected_child_node_children_index =
                if let Some(focus_action) = focus_actions.get(depth - 1) {
                    node.get_position_of_action(focus_action)
                        .ok_or_else(|| anyhow!("Focused action was not found"))?
                } else {
                    MCTS::<S, A, E, M, C, T, V>::select_path(node, &game_state, is_root, options)?
                };

            nodes_to_propagate_to_stack.push(NodeUpdateInfo {
                parent_node_index: latest_index,
                parent_node_player_to_move: game_engine.get_player_to_move(&game_state),
                node_child_index: selected_child_node_children_index,
            });

            let selected_child_node = &mut node.children[selected_child_node_children_index];

            game_state =
                Cow::Owned(game_engine.take_action(&game_state, selected_child_node.action()));
            move_number = game_engine.get_move_number(&game_state);

            let selected_child_node_node = &mut selected_child_node.node;
            let prev_visits = selected_child_node.visits;
            selected_child_node.visits += 1;
            node.visits += 1;

            if let Some(selected_child_node_index) = selected_child_node_node.get_index() {
                // If the node exists but visits was 0, then this node was cleared but the analysis was saved. Treat it as such by keeping the values.
                if prev_visits == 0 {
                    Self::update_node_values(
                        &nodes_to_propagate_to_stack,
                        selected_child_node_index,
                        move_number,
                        &mut *arena_mut,
                    );
                    break;
                }

                // Continue with the next iteration of the loop since we found an already expanded child node.
                latest_index = selected_child_node_index;
                continue;
            }

            // If the node is yet to be expanded and is not already expanded, then start expanding it.
            if selected_child_node_node.is_unexpanded() {
                selected_child_node_node.mark_as_expanding();
                drop(arena_mut);

                let expanded_node = Self::analyse_and_create_node(&game_state, analyzer).await;

                let mut arena_mut = arena.get_mut();
                let expanded_node_index = arena_mut.insert(expanded_node);

                let selected_child_node =
                    arena_mut.child(latest_index, selected_child_node_children_index);

                selected_child_node.node.set_expanded(expanded_node_index);

                Self::update_node_values(
                    &nodes_to_propagate_to_stack,
                    expanded_node_index,
                    move_number,
                    &mut *arena_mut,
                );

                break;
            }

            // The node must be expanding in another async operation. Wait for that to finish and continue the loop as if it was already expanded.
            let waiter = selected_child_node_node.get_waiter();
            drop(arena_mut);

            waiter.wait().await;

            latest_index = arena
                .get_mut()
                .child(latest_index, selected_child_node_children_index)
                .node
                .get_index()
                .expect("Node should have expanded");
        }

        Ok(depth)
    }

    fn update_node_values(
        nodes_to_update: &[NodeUpdateInfo],
        value_node_index: Index,
        value_node_move_num: usize,
        arena: &mut NodeArenaInner<MCTSNode<A, V>>,
    ) {
        let value_node = &arena.node(value_node_index);
        let value_score = &value_node.value_score.clone();
        let value_node_moves_left_score = value_node.moves_left_score;
        let value_node_game_length = value_node_move_num as f32 + value_node_moves_left_score;

        for NodeUpdateInfo {
            parent_node_index,
            node_child_index,
            parent_node_player_to_move,
        } in nodes_to_update
        {
            let node_to_update_parent = arena.node_mut(*parent_node_index);
            // Update value of W from the parent node's perspective.
            // This is because the parent chooses which child node to select, and as such will want the one with the
            // highest V from it's perspective. A node never cares what its value (W or Q) is from its own perspective.
            let score = value_score.get_value_for_player(*parent_node_player_to_move);

            let mut node_to_update = &mut node_to_update_parent.children[*node_child_index];
            node_to_update.W += score;
            node_to_update.M += value_node_game_length;
        }
    }

    fn _select_action(&mut self, use_temp: bool) -> Result<A> {
        if let Some(node_index) = &self.get_focus_node_index()? {
            let game_state = self.get_focus_node_game_state();
            let temp = &self.options.temperature;
            let temp = temp(&game_state);

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

    fn get_node_details(
        &self,
        node_index: Index,
        game_state: &S,
        is_root: bool,
    ) -> Result<NodeDetails<A>> {
        let arena_ref = self.arena.get();
        let node = arena_ref.node(node_index);

        let metrics =
            Self::get_PUCT_for_nodes(node, game_state, &*arena_ref, is_root, &self.options);

        let mut children: Vec<_> = node
            .children
            .iter()
            .zip(metrics)
            .map(|(n, m)| (n.action.clone(), m))
            .collect();

        children.sort_by(|(_, x_puct), (_, y_puct)| y_puct.cmp(x_puct));

        Ok(NodeDetails {
            visits: node.get_node_visits(),
            children,
        })
    }

    #[allow(clippy::await_holding_refcell_ref)]
    async fn advance_to_action_clearable(&mut self, action: A, clear: bool) -> Result<()> {
        self.clear_focus();

        let root_index = self.get_or_create_root_node().await;

        let game_engine = &self.game_engine;

        let mut arena_mut = self.arena.get_mut();
        let root_node = arena_mut.remove(root_index);
        let split_nodes = Self::split_node_children_by_action(&root_node, &action);

        if let Err(err) = split_nodes {
            // If there is an error, replace the root node back to it's original value.
            let index = arena_mut.insert(root_node);
            self.root = Some(index);
            return Err(err);
        }

        self.game_state = game_engine.take_action(&self.game_state, &action);

        let (chosen_node, other_nodes) = split_nodes.expect("Expected node to exist.");

        for node_index in other_nodes.into_iter().filter_map(|n| n.get_index()) {
            Self::remove_nodes_from_arena(node_index, &mut arena_mut);
        }
        drop(arena_mut);

        let chosen_node = if let Some(node_index) = chosen_node.get_index() {
            if clear {
                Self::clear_node(node_index, &mut self.arena.get_mut());
            }

            node_index
        } else {
            let node = Self::analyse_and_create_node(&self.game_state, self.analyzer).await;
            self.arena.get_mut().insert(node)
        };

        self.root.replace(chosen_node);

        Ok(())
    }

    fn remove_nodes_from_arena(node_index: Index, arena: &mut NodeArenaInner<MCTSNode<A, V>>) {
        let children = arena.remove(node_index).children;

        for child_node_index in children.into_iter().filter_map(|n| n.node.get_index()) {
            Self::remove_nodes_from_arena(child_node_index, arena);
        }
    }

    fn clear_node(node_index: Index, arena: &mut NodeArenaInner<MCTSNode<A, V>>) {
        let node = arena.node_mut(node_index);
        node.visits = 1;
        let children = &mut node.children;

        for child in children.iter_mut() {
            child.visits = 0;
            child.W = 0.0;
            child.M = 0.0;
        }

        let child_indexes: Vec<_> = children.iter().filter_map(|c| c.node.get_index()).collect();
        for child_index in child_indexes {
            Self::clear_node(child_index, arena);
        }
    }

    fn split_node_children_by_action<'b>(
        current_root: &'b MCTSNode<A, V>,
        action: &A,
    ) -> Result<(&'b MCTSNodeState, Vec<&'b MCTSNodeState>)> {
        let matching_action = current_root
            .get_child_of_action(action)
            .ok_or_else(|| anyhow!("No matching Action"))?;
        let other_actions: Vec<_> = current_root
            .children
            .iter()
            .filter(|n| n.action != *action)
            .map(|n| &n.node)
            .collect();

        Ok((&matching_action.node, other_actions))
    }

    fn select_path(
        node: &MCTSNode<A, V>,
        game_state: &S,
        is_root: bool,
        options: &MCTSOptions<S, C, T>,
    ) -> Result<usize> {
        let children = &node.children;
        let fpu = if is_root {
            options.fpu_root
        } else {
            options.fpu
        };
        let Nsb = node.get_node_visits();
        let root_Nsb = (Nsb as f32).sqrt();
        let cpuct = &options.cpuct;
        let cpuct = cpuct(game_state, Nsb, is_root);
        let game_length_baseline =
            &Self::get_game_length_baseline(children, options.moves_left_threshold);

        let mut best_child_index = 0;
        let mut best_puct = std::f32::MIN;

        for (i, child) in children.iter().enumerate() {
            let W = child.W;
            let Nsa = child.visits;
            let Psa = child.policy_score;
            let Usa = cpuct * Psa * root_Nsb / (1 + Nsa) as f32;
            let Qsa = if Nsa == 0 { fpu } else { W / Nsa as f32 };
            let logitQ = if options.logit_q { logit(Qsa) } else { Qsa };
            let Msa = Self::get_Msa(child, game_length_baseline, options);

            let PUCT = Msa + logitQ + Usa;

            if PUCT > best_puct {
                best_puct = PUCT;
                best_child_index = i;
            }
        }

        Ok(best_child_index)
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

    fn get_PUCT_for_nodes(
        node: &MCTSNode<A, V>,
        game_state: &S,
        arena: &NodeArenaInner<MCTSNode<A, V>>,
        is_root: bool,
        options: &MCTSOptions<S, C, T>,
    ) -> Vec<PUCT> {
        let children = &node.children;
        let fpu = if is_root {
            options.fpu_root
        } else {
            options.fpu
        };
        let Nsb = node.get_node_visits();
        let root_Nsb = (Nsb as f32).sqrt();
        let cpuct = &options.cpuct;
        let cpuct = cpuct(game_state, Nsb, is_root);
        let game_length_baseline =
            &Self::get_game_length_baseline(children, options.moves_left_threshold);

        let mut pucts = Vec::with_capacity(children.len());

        for child in children {
            let node = child.node.get_index().map(|index| arena.node(index));
            let W = child.W;
            let Nsa = child.visits;
            let Psa = child.policy_score;
            let Usa = cpuct * Psa * root_Nsb / (1 + Nsa) as f32;
            let Qsa = if Nsa == 0 { fpu } else { W / Nsa as f32 };
            let logitQ = if options.logit_q { logit(Qsa) } else { Qsa };
            let moves_left = node.map_or(0.0, |n| n.moves_left_score);
            let Msa = Self::get_Msa(child, game_length_baseline, options);
            let game_length = if Nsa == 0 {
                child.M
            } else {
                child.M / Nsa as f32
            };

            let PUCT = logitQ + Usa;
            pucts.push(PUCT {
                Psa,
                Nsa,
                Msa,
                cpuct,
                Usa,
                Qsa,
                logitQ,
                moves_left,
                game_length,
                PUCT,
            });
        }

        pucts
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

    async fn analyse_and_create_node(game_state: &S, analyzer: &M) -> MCTSNode<A, V> {
        analyzer.get_state_analysis(game_state).await.into()
    }

    fn apply_dirichlet_noise_to_node(node: &mut MCTSNode<A, V>, dirichlet: &DirichletOptions) {
        let policy_scores: Vec<f32> = node
            .children
            .iter()
            .map(|child_node| child_node.policy_score)
            .collect();

        let noisy_policy_scores = Self::generate_noise(policy_scores, dirichlet);

        for (child, policy_score) in node
            .children
            .iter_mut()
            .zip(noisy_policy_scores.into_iter())
        {
            child.policy_score = policy_score;
        }
    }

    fn get_Msa(
        child: &MCTSEdge<A>,
        game_length_baseline: &GameLengthBaseline,
        options: &MCTSOptions<S, C, T>,
    ) -> f32 {
        if child.visits == 0 {
            return 0.0;
        }

        if let GameLengthBaseline::None = game_length_baseline {
            return 0.0;
        }

        let (direction, game_length_baseline) =
            if let GameLengthBaseline::MinimizeGameLength(game_length_baseline) =
                game_length_baseline
            {
                (1.0f32, game_length_baseline)
            } else if let GameLengthBaseline::MaximizeGameLength(game_length_baseline) =
                game_length_baseline
            {
                (-1.0, game_length_baseline)
            } else {
                panic!();
            };

        let expected_game_length = child.M / child.visits as f32;
        let moves_left_scale = options.moves_left_scale;
        let moves_left_clamped = (game_length_baseline - expected_game_length)
            .min(moves_left_scale)
            .max(-moves_left_scale);
        let moves_left_scaled = moves_left_clamped / moves_left_scale;
        moves_left_scaled * options.moves_left_factor * direction
    }

    fn get_focus_node_index(&self) -> Result<Option<Index>> {
        let arena_ref = self.arena.get();

        let mut node_index = *self
            .root
            .as_ref()
            .ok_or_else(|| anyhow!("No root node found!"))?;

        for action in &self.focus_actions {
            match arena_ref
                .node(node_index)
                .get_child_of_action(action)
                .and_then(|child| child.node.get_index())
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

    fn get_game_length_baseline(
        edges: &[MCTSEdge<A>],
        moves_left_threshold: f32,
    ) -> GameLengthBaseline {
        if moves_left_threshold >= 1.0 {
            return GameLengthBaseline::None;
        }

        edges
            .iter()
            .max_by_key(|n| n.visits)
            .filter(|n| n.visits > 0)
            .map_or(GameLengthBaseline::None, |n| {
                let Qsa = n.W / n.visits as f32;
                let expected_game_length = n.M / n.visits as f32;

                if Qsa >= moves_left_threshold {
                    GameLengthBaseline::MinimizeGameLength(expected_game_length)
                } else if Qsa <= (1.0 - moves_left_threshold) {
                    GameLengthBaseline::MaximizeGameLength(expected_game_length)
                } else {
                    GameLengthBaseline::None
                }
            })
    }
}

fn logit(val: f32) -> f32 {
    if val <= 0.0 {
        -3.9855964
    } else if val >= 1.0 {
        3.9855964
    } else {
        (val / (1.0 - val)).ln() / 4.0
    }
}

enum GameLengthBaseline {
    MinimizeGameLength(f32),
    MaximizeGameLength(f32),
    None,
}

#[allow(non_snake_case)]
impl<A, V> MCTSNode<A, V> {
    pub fn new(
        value_score: V,
        policy_scores: Vec<ActionWithPolicy<A>>,
        moves_left_score: f32,
    ) -> Self {
        MCTSNode {
            visits: 1,
            value_score,
            moves_left_score,
            children: policy_scores
                .into_iter()
                .map(|action_with_policy| MCTSEdge {
                    visits: 0,
                    W: 0.0,
                    M: 0.0,
                    action: action_with_policy.action,
                    policy_score: action_with_policy.policy_score,
                    node: MCTSNodeState::Unexpanded,
                })
                .collect(),
        }
    }
}

impl<A, V> From<GameStateAnalysis<A, V>> for MCTSNode<A, V> {
    fn from(analysis: GameStateAnalysis<A, V>) -> Self {
        MCTSNode::new(
            analysis.value_score,
            analysis.policy_scores,
            analysis.moves_left,
        )
    }
}

impl<A, V> From<&MCTSNode<A, V>> for NodeMetrics<A, V>
where
    A: Clone,
    V: Clone,
{
    fn from(val: &MCTSNode<A, V>) -> Self {
        NodeMetrics {
            visits: val.visits,
            value: val.value_score.clone(),
            moves_left: val.moves_left_score,
            children: val.children.iter().map(|e| e.into()).collect_vec(),
        }
    }
}

impl<A> From<&MCTSEdge<A>> for NodeChildMetrics<A>
where
    A: Clone,
{
    fn from(val: &MCTSEdge<A>) -> Self {
        NodeChildMetrics::new(
            val.action.clone(),
            div_or_zero(val.W, val.visits as f32),
            div_or_zero(val.M, val.visits as f32),
            val.visits,
        )
    }
}

struct NodeUpdateInfo {
    parent_node_index: Index,
    node_child_index: usize,
    parent_node_player_to_move: usize,
}

#[allow(non_snake_case)]
#[derive(Debug)]
pub struct MCTSEdge<A> {
    action: A,
    W: f32,
    M: f32,
    visits: usize,
    policy_score: f32,
    node: MCTSNodeState,
}

impl<A> MCTSEdge<A> {
    pub fn visits(&self) -> usize {
        self.visits
    }

    pub fn node_index(&self) -> Option<Index> {
        self.node.get_index()
    }

    pub fn action(&self) -> &A {
        &self.action
    }
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

impl<A, V> NodeArenaInner<MCTSNode<A, V>> {
    #[inline]
    fn node(&self, index: Index) -> &MCTSNode<A, V> {
        &self.0[index]
    }

    #[inline]
    fn node_mut(&mut self, index: Index) -> &mut MCTSNode<A, V> {
        &mut self.0[index]
    }

    #[inline]
    fn child(&mut self, index: Index, child_index: usize) -> &mut MCTSEdge<A> {
        &mut self.0[index].children[child_index]
    }

    #[inline]
    fn insert(&mut self, node: MCTSNode<A, V>) -> Index {
        self.0.insert(node)
    }

    #[inline]
    fn remove(&mut self, index: Index) -> MCTSNode<A, V> {
        self.0
            .remove(index)
            .expect("Expected node to exist in the arena")
    }
}

#[derive(Debug)]
pub struct MCTSNode<A, V> {
    visits: usize,
    value_score: V,
    moves_left_score: f32,
    children: Vec<MCTSEdge<A>>,
}

impl<A, V> MCTSNode<A, V>
where
    A: Eq,
{
    pub fn get_node_visits(&self) -> usize {
        self.visits
    }

    pub fn get_child_of_action(&self, action: &A) -> Option<&MCTSEdge<A>> {
        self.iter_edges().find(|c| c.action == *action)
    }

    pub fn get_position_of_action(&self, action: &A) -> Option<usize> {
        self.iter_edges().position(|c| c.action == *action)
    }

    pub fn is_terminal(&self) -> bool {
        self.children.is_empty()
    }

    pub fn iter_edges(&self) -> impl Iterator<Item = &MCTSEdge<A>> {
        self.children.iter()
    }
}

#[derive(Debug)]
enum MCTSNodeState {
    Unexpanded,
    Expanding,
    ExpandingWithWaiters(Rc<LocalManualResetEvent>),
    Expanded(Index),
}

impl MCTSNodeState {
    fn get_index(&self) -> Option<Index> {
        if let Self::Expanded(index) = self {
            Some(*index)
        } else {
            None
        }
    }

    fn is_unexpanded(&self) -> bool {
        matches!(self, Self::Unexpanded)
    }

    fn mark_as_expanding(&mut self) {
        debug_assert!(matches!(self, Self::Unexpanded));
        *self = Self::Expanding
    }

    fn set_expanded(&mut self, index: Index) {
        debug_assert!(!matches!(self, Self::Unexpanded));
        debug_assert!(!matches!(self, Self::Expanded(_)));
        let state = std::mem::replace(self, Self::Expanded(index));
        if let Self::ExpandingWithWaiters(reset_events) = state {
            reset_events.set()
        }
    }

    fn get_waiter(&mut self) -> Rc<LocalManualResetEvent> {
        match self {
            Self::Expanding => {
                let reset_event = Rc::new(LocalManualResetEvent::new(false));
                *self = Self::ExpandingWithWaiters(reset_event.clone());
                reset_event
            }
            Self::ExpandingWithWaiters(reset_event) => reset_event.clone(),
            _ => panic!("Node state is not currently expanding"),
        }
    }
}
