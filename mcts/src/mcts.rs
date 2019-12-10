use std::cell::{Cell,RefCell};
use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::{Arc};
use futures::stream::{FuturesOrdered,StreamExt};
use async_std::sync::{RwLock};
use generational_arena::{Arena,Index};
use rand::{thread_rng,Rng};
use rand::prelude::Distribution;
use rand::distributions::WeightedIndex;
use rand_distr::Dirichlet;
use failure::{Error,format_err};

use engine::game_state::GameState;
use engine::value::Value;
use engine::engine::{GameEngine};
use model::analytics::{ActionWithPolicy,GameAnalyzer};
use model::node_metrics::{NodeMetrics};
use super::node_details::{PUCT,NodeDetails};

pub struct DirichletOptions {
    pub alpha: f32,
    pub epsilon: f32
}

pub struct MCTSOptions<S,C,T>
where
    S: GameState,
    C: Fn(&S, usize, usize, bool) -> f32,
    T: Fn(&S, usize) -> f32
{
    dirichlet: Option<DirichletOptions>,
    fpu: f32,
    fpu_root: f32,
    cpuct: C,
    temperature: T,
    temperature_visit_offset: f32,
    parallelism: usize,
    _phantom_state: PhantomData<*const S>
}

impl<S,C,T> MCTSOptions<S,C,T>
where
    S: GameState,
    C: Fn(&S, usize, usize, bool) -> f32,
    T: Fn(&S, usize) -> f32
{
    pub fn new(
        dirichlet: Option<DirichletOptions>,
        fpu: f32,
        fpu_root: f32,
        cpuct: C,
        temperature: T,
        temperature_visit_offset: f32,
        parallelism: usize
    ) -> Self {
        MCTSOptions {
            dirichlet,
            fpu,
            fpu_root,
            cpuct,
            temperature,
            temperature_visit_offset,
            parallelism,
            _phantom_state: PhantomData
        }
    }
}

pub struct MCTS<'a,S,A,E,M,C,T,V>
where
    S: GameState,
    A: Clone + Eq + Debug,
    V: Value,
    E: GameEngine,
    M: GameAnalyzer,
    C: Fn(&S, usize, usize, bool) -> f32,
    T: Fn(&S, usize) -> f32
{
    options: MCTSOptions<S,C,T>,
    game_engine: &'a E,
    analyzer: &'a M,
    starting_game_state: Option<S>,
    starting_num_actions: Option<usize>,
    root: Option<Index>,
    arena: RefCell<Arena<MCTSNode<S,A,V>>>
}

#[allow(non_snake_case)]
#[derive(Debug)]
struct MCTSNode<S,A,V> {
    value_score: V,
    W: Cell<f32>,
    game_state: S,
    num_actions: usize,
    children: Vec<MCTSChildNode<A>>
}

#[derive(Debug)]
enum MCTSNodeState {
    Unexpanded,
    // @TODO: The RwLock is to act as a notification system that the node is ready and expanded. Then the awaits know to continue. There must be a better way to achieve this.
    Expanding(Box<Arc<RwLock<()>>>),
    Expanded(Index)
}

#[derive(Debug)]
struct MCTSChildNode<A> {
    action: A,
    visits: Cell<usize>,
    in_flight: Cell<usize>,
    policy_score: f32,
    state: MCTSNodeState
}

#[derive(Debug)]
struct NodePUCT<'a, A> {
    node: &'a MCTSChildNode<A>,
    score: f32
}

#[allow(non_snake_case)]
impl<'a,S,A,E,M,C,T,V> MCTS<'a,S,A,E,M,C,T,V>
where
    S: GameState,
    A: Clone + Eq + Debug,
    V: Value,
    E: 'a + GameEngine<State=S,Action=A,Value=V>,
    M: 'a + GameAnalyzer<State=S,Action=A,Value=V>,
    C: Fn(&S, usize, usize, bool) -> f32,
    T: Fn(&S, usize) -> f32
{
    pub fn new(
        game_state: S,
        actions: usize,
        game_engine: &'a E,
        analyzer: &'a M,
        options: MCTSOptions<S,C,T>
    ) -> Self {
        MCTS {
            options,
            game_engine,
            analyzer,
            starting_game_state: Some(game_state),
            starting_num_actions: Some(actions),
            root: None,
            arena: RefCell::new(Arena::new())
        }
    }

    pub fn with_capacity(
        game_state: S,
        actions: usize,
        game_engine: &'a E,
        analyzer: &'a M,
        options: MCTSOptions<S,C,T>,
        capacity: usize
    ) -> Self {
        MCTS {
            options,
            game_engine,
            analyzer,
            starting_game_state: Some(game_state),
            starting_num_actions: Some(actions),
            root: None,
            arena: RefCell::new(Arena::with_capacity(capacity * 2))
        }
    }

    pub async fn search(&mut self, visits: usize) -> Result<usize, Error> {
        self._search(visits, true).await
    }

    pub async fn search_no_noise(&mut self, visits: usize) -> Result<usize, Error> {
        self._search(visits, false).await
    }

    pub async fn select_action(&mut self) -> Result<A, Error> {
        self._select_action(true).await
    }

    pub async fn select_action_no_temp(&mut self) -> Result<A, Error> {
        self._select_action(false).await
    }

    pub async fn advance_to_action(&mut self, action: A) -> Result<(), Error> {
        self.advance_to_action_clearable(action, true).await
    }

    pub async fn advance_to_action_retain(&mut self, action: A) -> Result<(), Error> {
        self.advance_to_action_clearable(action, false).await
    }

    pub async fn get_root_node_metrics(&mut self) -> Result<NodeMetrics<A>, Error> {
        let root_index = self.root.ok_or(format_err!("No root node found!"))?;
        let root = &self.arena.borrow()[root_index];

        Ok(NodeMetrics {
            visits: root.get_node_visits(),
            W: root.W.get(),
            children_visits: root.children.iter().map(|n| (
                n.action.clone(),
                n.visits.get()
            )).collect()
        })
    }

    pub async fn get_root_node_details(&self) -> Result<NodeDetails<A>, Error> {
        let root_index = self.root.as_ref().ok_or(format_err!("No root node found!"))?;
        self.get_node_details(*root_index, true).await
    }

    pub async fn get_principal_variation(&self) -> Result<Vec<(A, PUCT)>, Error> {
        let arena_borrow = &self.arena.borrow();

        let mut node_index = *self.root.as_ref().ok_or(format_err!("No root node found!"))?;
        let mut nodes = vec!();

        loop {
            let is_root = nodes.len() == 0;
            let mut children = self.get_node_details(node_index, is_root).await?.children;

            if children.len() == 0 { break; }

            let (action, puct) = children.swap_remove(0);
            nodes.push((action.clone(), puct));

            if let Some(child) = arena_borrow[node_index].get_child_of_action(&action) {
                if let Some(child_index) = child.state.get_index() {
                    node_index = child_index;
                    continue;
                }
            }

            break;
        }

        Ok(nodes)
    }

    async fn _search(&mut self, visits: usize, apply_noise: bool) -> Result<usize, Error> {
        let game_engine = &self.game_engine;
        let fpu = self.options.fpu;
        let fpu_root = self.options.fpu_root;
        let cpuct = &self.options.cpuct;
        let dirichlet = if apply_noise { &self.options.dirichlet } else { &None };
        let analyzer = &mut self.analyzer;
        let root = &mut self.root;
        let starting_num_actions = &mut self.starting_num_actions;
        let starting_game_state = &mut self.starting_game_state;
        let arena_cell = &self.arena;

        let mut arena_borrow_mut = arena_cell.borrow_mut();
        let root_node_index = MCTS::<S,A,E,M,C,T,V>::get_or_create_root_node(
            root,
            starting_game_state,
            starting_num_actions,
            analyzer,
            &mut *arena_borrow_mut,
            dirichlet
        ).await;

        let current_visits = arena_borrow_mut[root_node_index].get_node_visits();
        drop(arena_borrow_mut);

        let mut max_depth: usize = 0;
        let mut searches_remaining = visits - current_visits;
        let initial_searches = self.options.parallelism.min(searches_remaining);

        let mut searches = FuturesOrdered::new();
        for _ in 0..initial_searches {
            searches_remaining -= 1;
            let future = recurse_path_and_expand::<S,A,E,M,C,T,V>(root_node_index, arena_cell, game_engine, analyzer, fpu, fpu_root, cpuct);

            searches.push(future);
        }

        while let Some(search_depth) = searches.next().await {
            if searches_remaining > 0 {
                searches_remaining -= 1;
                let future = recurse_path_and_expand::<S,A,E,M,C,T,V>(root_node_index, arena_cell, game_engine, analyzer, fpu, fpu_root, cpuct);
                searches.push(future);
            }

            max_depth = max_depth.max(search_depth?);
        }

        Ok(max_depth)
    }

    async fn _select_action(&mut self, use_temp: bool) -> Result<A, Error> {
        if let Some(root_node_index) = &self.root {
            let arena_borrow = self.arena.borrow();
            let root_node = &arena_borrow[*root_node_index];
            let temp = &self.options.temperature;
            let temperature_visit_offset = self.options.temperature_visit_offset;
            let game_state = &root_node.game_state;
            let temp = temp(game_state, root_node.num_actions);
            drop(arena_borrow);
            let child_node_details = self.get_root_node_details().await?.children;

            if child_node_details.len() == 0 {
                let arena_borrow = self.arena.borrow();
                let root_node = &arena_borrow[*root_node_index];
                let game_state = &root_node.game_state;
                return Err(format_err!("Node has no children. This node should have been designated as a terminal node. {:?}", game_state));
            }

            let best_action = if temp == 0.0 || !use_temp {
                let (best_action, _) = child_node_details.first().ok_or_else(|| format_err!("No available actions"))?;
                best_action
            } else {
                let candidates: Vec<_> = child_node_details.iter().map(|(a, puct)| (a, puct.Nsa)).collect();
                let chosen_index = Self::select_action_using_temperature(&candidates, temp, temperature_visit_offset)?;
                candidates[chosen_index].0
            };

            return Ok(best_action.clone());
        }

        return Err(format_err!("Root node does not exist. Run search first."));
    }

    async fn get_node_details(&self, node_index: Index, is_root: bool) -> Result<NodeDetails<A>, Error> {
        let arena_borrow = &self.arena.borrow();
        let root = &arena_borrow[node_index];
        let root_visits = root.get_node_visits();
        let children = &root.children;
        let options = &self.options;

        let metrics = Self::get_PUCT_for_nodes(
            children,
            &*arena_borrow,
            root_visits,
            &root.game_state,
            is_root,
            root.num_actions,
            options.fpu,
            options.fpu_root,
            &options.cpuct
        );

        let mut children: Vec<_> = children.iter().zip(metrics).map(|(n, m)| (
            n.action.clone(),
            m
        )).collect();

        children.sort_by(|(_, x_puct), (_, y_puct)| y_puct.cmp(&x_puct));

        Ok(NodeDetails {
            visits: root_visits,
            W: root.W.get(),
            children
        })
    }

    async fn advance_to_action_clearable(&mut self, action: A, clear: bool) -> Result<(), Error> {
        let mut root = self.root.take();
        let analyzer = &mut self.analyzer;
        let starting_game_state = &mut self.starting_game_state;
        let starting_num_actions = &mut self.starting_num_actions;
        let game_engine = &self.game_engine;
        let dirichlet = &self.options.dirichlet;

        let arena_borrow_mut = &mut *self.arena.borrow_mut();
        let root_index = MCTS::<S,A,E,M,C,T,V>::get_or_create_root_node(&mut root, starting_game_state, starting_num_actions, analyzer, arena_borrow_mut, dirichlet).await;

        let root_node = arena_borrow_mut.remove(root_index).expect("Root node should exist in arena.");
        let split_nodes = Self::split_node_children_by_action(&root_node, &action);

        if let Err(err) = split_nodes {
            // If there is an error, replace the root node back to it's original value.
            let index = arena_borrow_mut.insert(root_node);
            self.root = Some(index);
            return Err(err);
        }

        let (chosen_node, other_nodes) = split_nodes.unwrap();

        for node_index in other_nodes.into_iter().filter_map(|n| n.get_index()) {
            Self::remove_nodes_from_arena(node_index, arena_borrow_mut);
        }

        let chosen_node = if let Some(node_index) = chosen_node.get_index() {
            if clear {
                Self::clear_node_visits(node_index, arena_borrow_mut);
            } else {
                // Always clear the node's W since it is no longer relevant
                let node = &mut arena_borrow_mut[node_index];
                node.W.set(0.0);
            }

            node_index
        } else {
            let prior_num_actions = root_node.num_actions;
            let new_game_state = game_engine.take_action(&root_node.game_state, &action);
            let node = MCTS::<S,A,E,M,C,T,V>::expand_leaf(new_game_state, prior_num_actions, analyzer).await;
            arena_borrow_mut.insert(node)
        };

        self.root.replace(chosen_node);

        Ok(())
    }

    fn remove_nodes_from_arena(node_index: Index, arena: &mut Arena<MCTSNode<S,A,V>>) {
        let child_node = arena.remove(node_index).unwrap();

        for child_node_index in child_node.children.into_iter().filter_map(|n| n.state.get_index()) {
            Self::remove_nodes_from_arena(child_node_index, arena);
        }
    }

    fn clear_node_visits(node_index: Index, arena: &mut Arena<MCTSNode<S,A,V>>) {
        let node = &mut arena[node_index];
        node.W.set(0.0);

        for child in &node.children {
            child.visits.set(0);
        }

        let child_indexes: Vec<_> = node.children.iter().filter_map(|c| c.state.get_index()).collect();
        for child_index in child_indexes {
            Self::clear_node_visits(child_index, arena);
        }
    }

    fn split_node_children_by_action<'b>(current_root: &'b MCTSNode<S,A,V>, action: &A) -> Result<(&'b MCTSNodeState, Vec<&'b MCTSNodeState>), Error> {
        let matching_action = current_root.get_child_of_action(action).ok_or(format_err!("No matching Action"))?;
        let other_actions: Vec<_> = current_root.children.iter().filter(|n| n.action != *action).map(|n| &n.state).collect();

        Ok((&matching_action.state, other_actions))
    }

    fn select_path<'b>(nodes: &'b [MCTSChildNode<A>], arena: &Arena<MCTSNode<S,A,V>>, Nsb: usize, game_state: &S, is_root: bool, prior_num_actions: usize, fpu: f32, fpu_root: f32, cpuct: &C) -> Result<&'b MCTSChildNode<A>, Error> {
        let fpu = if is_root { fpu_root } else { fpu };
        let root_Nsb = (Nsb as f32).sqrt();
        let mut best_node = &nodes[0];
        let mut best_puct = std::f32::MIN;
        let cpuct = cpuct(game_state, prior_num_actions, Nsb, is_root);

        for child in nodes {
            let W = child.state.get_index().map_or(0.0, |i| arena[i].W.get());
            let Nsa = child.visits.get();
            let Psa = child.policy_score;
            let Usa = cpuct * Psa * root_Nsb / (1 + Nsa) as f32;
            let virtual_loss = child.in_flight.get() as f32;
            let Qsa = if Nsa == 0 { fpu } else { (W - virtual_loss) / (Nsa as f32 + virtual_loss) };

            let PUCT = Qsa + Usa;

            if PUCT > best_puct {
                best_puct = PUCT;
                best_node = child;
            }
        }

        Ok(best_node)
    }

    fn select_action_using_temperature(action_visits: &[(&A, usize)], temp: f32, temperature_visit_offset: f32) -> Result<usize, Error> {
        let normalized_visits = action_visits.iter().map(|(_, visits)| (*visits as f32 + temperature_visit_offset).min(0.0).powf(1.0 / temp));

        let weighted_index = WeightedIndex::new(normalized_visits);

        let chosen_idx = match weighted_index {
            Err(_) => {
                println!("Invalid puct scores. Most likely all are 0. Move will be randomly selected.");
                println!("{:?}", action_visits);
                thread_rng().gen_range(0, action_visits.len())
            },
            Ok(weighted_index) => weighted_index.sample(&mut thread_rng())
        };

        Ok(chosen_idx)
    }

    fn get_PUCT_for_nodes(nodes: &[MCTSChildNode<A>], arena: &Arena<MCTSNode<S,A,V>>, Nsb: usize, game_state: &S, is_root: bool, prior_num_actions: usize, fpu: f32, fpu_root: f32, cpuct: &C) -> Vec<PUCT>
    {
        let fpu = if is_root { fpu_root } else { fpu };
        let mut pucts = Vec::with_capacity(nodes.len());
        let root_Nsb = (Nsb as f32).sqrt();
        let cpuct = cpuct(game_state, prior_num_actions, Nsb, is_root);

        for child in nodes {
            let W = child.state.get_index().map_or(0.0, |i| arena[i].W.get());
            let Nsa = child.visits.get();
            let Psa = child.policy_score;
            let Usa = cpuct * Psa * root_Nsb / (1 + Nsa) as f32;
            let virtual_loss = child.in_flight.get() as f32;
            let Qsa = if Nsa == 0 { fpu } else { (W - virtual_loss) / (Nsa as f32 + virtual_loss) };

            let PUCT = Qsa + Usa;
            pucts.push(PUCT { Psa, Nsa, cpuct, Usa, Qsa, PUCT });
        }

        pucts
    }

    async fn expand_leaf(game_state: S, prior_num_actions: usize, analyzer: &M) -> MCTSNode<S,A,V> {
        let num_actions = prior_num_actions + 1;
        MCTS::<S,A,E,M,C,T,V>::analyse_and_create_node(game_state, num_actions, analyzer).await
    }

    async fn get_or_create_root_node(
        root: &mut Option<Index>,
        starting_game_state: &mut Option<S>,
        starting_num_actions: &mut Option<usize>,
        analyzer: &M,
        arena: &mut Arena<MCTSNode<S,A,V>>,
        dirichlet: &Option<DirichletOptions>
    ) -> Index {
        if let Some(root_node_index) = root.as_ref() {
            return *root_node_index;
        }

        let starting_game_state = starting_game_state.take().expect("Tried to use the same starting game state twice");
        let starting_num_actions = starting_num_actions.take().expect("Tried to use the same starting actions twice");

        let mut root_node = MCTS::<S,A,E,M,C,T,V>::analyse_and_create_node(
            starting_game_state,
            starting_num_actions,
            analyzer
        ).await;

        Self::apply_dirichlet_noise_to_node(&mut root_node, dirichlet);

        let root_node_index = arena.insert(root_node);

        root.replace(root_node_index);

        root_node_index
    }

    async fn analyse_and_create_node(game_state: S, actions: usize, analyzer: &M) -> MCTSNode<S,A,V> {
        let analysis_result = analyzer.get_state_analysis(&game_state).await;
        let value_score = analysis_result.value_score;

        MCTSNode::new(game_state, actions, value_score, analysis_result.policy_scores)
    }

    fn apply_dirichlet_noise_to_node(node: &mut MCTSNode<S,A,V>, dirichlet: &Option<DirichletOptions>) {
        if let Some(dirichlet) = dirichlet {
            let policy_scores: Vec<f32> = node.children.iter().map(|child_node| {
                child_node.policy_score
            }).collect();

            let updated_policy_scores = MCTS::<S,A,E,M,C,T,V>::apply_dirichlet_noise(policy_scores, dirichlet);

            for (child, policy_score) in node.children.iter_mut().zip(updated_policy_scores.into_iter()) {
                child.policy_score = policy_score;
            }
        }
    }

    fn apply_dirichlet_noise(policy_scores: Vec<f32>, dirichlet: &DirichletOptions) -> Vec<f32>
    {
        // Do not apply noise if there is only one action.
        if policy_scores.len() < 2 {
            return policy_scores;
        }

        let e = dirichlet.epsilon;
        let dirichlet_noise = Dirichlet::new_with_size(dirichlet.alpha, policy_scores.len())
            .expect("Error creating dirichlet distribution")
            .sample(&mut thread_rng());

        dirichlet_noise.into_iter().zip(policy_scores).map(|(noise, policy_score)|
            (1.0 - e) * policy_score + e * noise
        ).collect()
    }
}

#[allow(non_snake_case)]
impl<S,A,V> MCTSNode<S,A,V> {
    pub fn new(game_state: S, num_actions: usize, value_score: V, policy_scores: Vec<ActionWithPolicy<A>>) -> Self {
        MCTSNode {
            value_score,
            W: Cell::new(0.0),
            game_state,
            num_actions,
            children: policy_scores.into_iter().map(|action_with_policy| {
                MCTSChildNode {
                    visits: Cell::new(0),
                    in_flight: Cell::new(0),
                    action: action_with_policy.action,
                    policy_score: action_with_policy.policy_score,
                    state: MCTSNodeState::Unexpanded
                }
            }).collect()
        }
    }
}

#[allow(non_snake_case)]
async fn recurse_path_and_expand<'a,S,A,E,M,C,T,V>(
    root_index: Index,
    arena: &RefCell<Arena<MCTSNode<S,A,V>>>,
    game_engine: &E,
    analyzer: &M,
    fpu: f32,
    fpu_root: f32,
    cpuct: &C
) -> Result<usize, Error>
where
    S: GameState,
    A: Clone + Eq + Debug,
    V: Value,
    E: GameEngine<State=S,Action=A,Value=V>,
    M: GameAnalyzer<State=S,Action=A,Value=V>,
    C: Fn(&S, usize, usize, bool) -> f32,
    T: Fn(&S, usize) -> f32
{
    let mut depth = 0;
    let mut node_stack = vec!(root_index);
    let mut in_flight_stack = vec!();

    'outer: loop {
        depth += 1;
        if let Some(latest_index) = node_stack.last() {
            let arena_borrow = arena.borrow();
            let node = &arena_borrow[*latest_index];

            // If the node is a terminal node.
            let children = &node.children;
            if children.len() == 0 {
                node_stack.pop();
                update_Ws(node_stack, &node, &arena_borrow, game_engine);
                break 'outer;
            }

            let game_state = &node.game_state;
            let prior_num_actions = node.num_actions;
            let is_root = depth == 1;
            let Nsb = node.get_node_visits();

            let selected_child_node = MCTS::<S,A,E,M,C,T,V>::select_path(
                children,
                &*arena_borrow,
                Nsb,
                game_state,
                is_root,
                prior_num_actions,
                fpu,
                fpu_root,
                cpuct
            )?;

            let prev_visits = selected_child_node.visits.get();
            selected_child_node.visits.set(prev_visits + 1);
            selected_child_node.in_flight.set(selected_child_node.in_flight.get() + 1);
            in_flight_stack.push((*latest_index, selected_child_node.action.clone()));

            if let MCTSNodeState::Expanded(selected_child_node_index) = selected_child_node.state {
                let node = &arena_borrow[selected_child_node_index];
                // If the node exists but visits was 0, then this node was cleared but the analysis was saved. Treat it as such by keeping the values.
                if prev_visits == 0 {
                    update_Ws(node_stack, &node, &arena_borrow, game_engine);
                    break 'outer;
                }

                // Continue with the next iteration of the loop since we found an already expanded child node.
                node_stack.push(selected_child_node_index);
                continue 'outer;
            }

            let selected_action = selected_child_node.action.clone();
            drop(arena_borrow);

            'inner: loop {
                let arena_borrow = arena.borrow();
                let prior_node = &arena_borrow[*latest_index];
                let selected_child_node = prior_node.get_child_of_action(&selected_action).unwrap();
                let selected_child_node_state = &selected_child_node.state;
                if let MCTSNodeState::Expanded(selected_child_node_index) = selected_child_node.state {
                    node_stack.push(selected_child_node_index);
                    continue 'outer;
                }

                // If the node is currently expanding. Wait for the expansion to be completed.
                if let MCTSNodeState::Expanding(expanding_lock) = selected_child_node_state {
                    let expanding_lock = expanding_lock.clone();
                    drop(arena_borrow);

                    expanding_lock.read().await;

                    continue 'inner;
                }

                // It is not expanded or expanded so let's expand it.
                drop(arena_borrow);
                let mut arena_borrow_mut = arena.borrow_mut();
                let prior_node = &mut arena_borrow_mut[*latest_index];
                let selected_child_node = prior_node.get_child_of_action_mut(&selected_action).unwrap();
                let selected_child_node_state = &mut selected_child_node.state;

                // Double check that the node has not changed now that the lock has been reacquired as a write.
                if let MCTSNodeState::Unexpanded = selected_child_node_state {
                    // Immediately replace the state with an indication that we are expanding.RwLock
                    let expanding_lock = std::sync::Arc::new(RwLock::new(()));
                    std::mem::replace(selected_child_node_state, MCTSNodeState::Expanding(Box::new(expanding_lock.clone())));

                    let new_game_state = game_engine.take_action(&prior_node.game_state, &selected_action);
                    let prior_num_actions = prior_node.num_actions;

                    drop(arena_borrow_mut);

                    let expanding_write_lock = expanding_lock.write().await;

                    let expanded_node = MCTS::<S,A,E,M,C,T,V>::expand_leaf(
                        new_game_state,
                        prior_num_actions,
                        analyzer
                    ).await;

                    let mut arena_borrow_mut = arena.borrow_mut();
                    let latest_index = *latest_index;
                    update_Ws(node_stack, &expanded_node, &arena_borrow_mut, game_engine);

                    update_child_with_expanded_node(latest_index, expanded_node, &selected_action, &mut arena_borrow_mut);
                    drop(expanding_write_lock);

                    break 'outer;
                }
            }
        }
    }

    let arena_borrow = arena.borrow();

    for (parent_node_index, action) in in_flight_stack {
        let parent_node = &arena_borrow[parent_node_index];
        let in_flight = &parent_node.get_child_of_action(&action).unwrap().in_flight;
        in_flight.set(in_flight.get() - 1);
    }

    Ok(depth)
}

fn update_child_with_expanded_node<S,A,V>(prior_index: Index, expanded_node: MCTSNode<S,A,V>, action: &A, arena: &mut Arena<MCTSNode<S,A,V>>)
where
    A: Eq
{
    let expanded_node_index = arena.insert(expanded_node);
    let prior_node = &mut arena[prior_index];
    let selected_child_node = prior_node.get_child_of_action_mut(action).unwrap();
    let selected_child_node_state = &mut selected_child_node.state;
    std::mem::replace(selected_child_node_state, MCTSNodeState::Expanded(expanded_node_index));
}

#[allow(non_snake_case)]
fn update_Ws<S,A,V,E>(nodes: Vec<Index>, value_node: &MCTSNode<S,A,V>, arena: &Arena<MCTSNode<S,A,V>>, game_engine: &E)
where
    V: Value,
    E: GameEngine<State=S,Action=A,Value=V>
{
    let value_score = &value_node.value_score;
    let mut nodes: Vec<_> = nodes.into_iter().map(|node_index| &arena[node_index]).collect();
    nodes.push(value_node);

    for (parent_node, child_node) in nodes.iter().zip(nodes.iter().skip(1)) {
        let W = &child_node.W;
        // Update value of W from the parent node's perspective.
        // This is because the parent chooses which child node to select, and as such will want the one with the
        // highest V from it's perspective. A node never cares what its value (W or Q) is from its own perspective.
        let player_to_move = game_engine.get_player_to_move(&parent_node.game_state);
        let score = value_score.get_value_for_player(player_to_move);
        W.set(W.get() + score);
    }
}

impl<S,A,V> MCTSNode<S,A,V>
where
    A: Eq
{
    fn get_node_visits(&self) -> usize {
        self.children.iter().map(|c| c.visits.get()).sum::<usize>() + 1
    }

    fn get_child_of_action(&self, action: &A) -> Option<&MCTSChildNode<A>> {
        self.children.iter().find(|c| c.action == *action)
    }

    fn get_child_of_action_mut(&mut self, action: &A) -> Option<&mut MCTSChildNode<A>> {
        self.children.iter_mut().find(|c| c.action == *action)
    }
}

impl MCTSNodeState {
    fn get_index(&self) -> Option<Index> {
        if let Self::Expanded(index) = self { Some(*index) } else { None }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::counting_game::{CountingAction,CountingAnalyzer,CountingGameEngine,CountingGameState};
    use assert_approx_eq::assert_approx_eq;

    const ERROR_DIFF: f32 = 0.02;

    fn assert_metrics(left: &NodeMetrics<CountingAction>, right: &NodeMetrics<CountingAction>) {
        assert_eq!(left.W, right.W);
        assert_eq!(left.visits, right.visits);
        assert_eq!(left.children_visits.len(), right.children_visits.len());

        for ((l_a, l_visits), (r_a, r_visits)) in left.children_visits.iter().zip(right.children_visits.iter()) {
            assert_eq!(l_a, r_a);
            let allowed_diff = ((*l_visits).max(*r_visits) as f32) * ERROR_DIFF + 0.9;
            assert_approx_eq!(*l_visits as f32, *r_visits as f32, allowed_diff);
        }
    }

    #[tokio::test]
    async fn test_mcts_is_deterministic() {
        let game_state = CountingGameState::initial();
        let actions = 0;
        let game_engine = CountingGameEngine::new();
        let analyzer = CountingAnalyzer::new();

        let mut mcts = MCTS::new(game_state.to_owned(), actions, &game_engine, &analyzer, MCTSOptions::new(
            None,
            0.0,
            0.0,
            |_,_,_,_| 1.0,
            |_,_| 0.0,
            0.0,
            1
        ));

        let mut mcts2 = MCTS::new(game_state, actions, &game_engine, &analyzer, MCTSOptions::new(
            None,
            0.0,
            0.0,
            |_,_,_,_| 1.0,
            |_,_| 0.0,
            0.0,
            1
        ));

        mcts.search(800).await.unwrap();
        mcts2.search(800).await.unwrap();

        let metrics = mcts.get_root_node_metrics().await.unwrap();
        let metrics2 = mcts.get_root_node_metrics().await.unwrap();

        assert_metrics(&metrics, &metrics2);
    }

    #[tokio::test]
    async fn test_mcts_chooses_best_p1_move() {
        let game_state = CountingGameState::initial();
        let actions = 0;
        let game_engine = CountingGameEngine::new();
        let analyzer = CountingAnalyzer::new();

        let mut mcts = MCTS::new(game_state, actions, &game_engine, &analyzer, MCTSOptions::new(
            None,
            0.0,
            0.0,
            |_,_,_,_| 1.0,
            |_,_| 0.0,
            0.0,
            1
        ));

        mcts.search(800).await.unwrap();
        let action = mcts.select_action().await.unwrap();

        assert_eq!(action, CountingAction::Increment);
    }

    #[tokio::test]
    async fn test_mcts_chooses_best_p2_move() {
        let game_state = CountingGameState::initial();
        let actions = 0;
        let game_engine = CountingGameEngine::new();
        let analyzer = CountingAnalyzer::new();

        let mut mcts = MCTS::new(game_state, actions, &game_engine, &analyzer, MCTSOptions::new(
            None,
            0.0,
            0.0,
            |_,_,_,_| 1.0,
            |_,_| 0.0,
            0.0,
            1
        ));

        mcts.advance_to_action(CountingAction::Increment).await.unwrap();

        mcts.search(800).await.unwrap();
        let action = mcts.select_action().await.unwrap();

        assert_eq!(action, CountingAction::Decrement);
    }

    #[tokio::test]
    async fn test_mcts_advance_to_next_works_without_search() {
        let game_state = CountingGameState::initial();
        let actions = 0;
        let game_engine = CountingGameEngine::new();
        let analyzer = CountingAnalyzer::new();

        let mut mcts = MCTS::new(game_state, actions, &game_engine, &analyzer, MCTSOptions::new(
            None,
            0.0,
            0.0,
            |_,_,_,_| 1.0,
            |_,_| 0.0,
            0.0,
            1
        ));

        mcts.advance_to_action(CountingAction::Increment).await.unwrap();
    }

    #[tokio::test]
    async fn test_mcts_metrics_returns_accurate_results() {
        let game_state = CountingGameState::initial();
        let actions = 0;
        let game_engine = CountingGameEngine::new();
        let analyzer = CountingAnalyzer::new();

        let mut mcts = MCTS::new(game_state, actions, &game_engine, &analyzer, MCTSOptions::new(
            None,
            0.0,
            0.0,
            |_,_,_,_| 1.0,
            |_,_| 0.0,
            0.0,
            1
        ));

        mcts.search(800).await.unwrap();

        let metrics = mcts.get_root_node_metrics().await.unwrap();

        assert_metrics(&metrics, &NodeMetrics {
            visits: 800,
            W: 0.0,
            children_visits: vec!(
                (CountingAction::Increment, 316),
                (CountingAction::Decrement, 179),
                (CountingAction::Stay, 304)
            )
        });
    }

    #[tokio::test]
    async fn test_mcts_weights_policy_initially() {
        let game_state = CountingGameState::initial();
        let actions = 0;
        let game_engine = CountingGameEngine::new();
        let analyzer = CountingAnalyzer::new();

        let mut mcts = MCTS::new(game_state, actions, &game_engine, &analyzer, MCTSOptions::new(
            None,
            0.0,
            0.0,
            |_,_,_,_| 1.0,
            |_,_| 0.0,
            0.0,
            1
        ));

        mcts.search(100).await.unwrap();

        let metrics = mcts.get_root_node_metrics().await.unwrap();

        assert_metrics(&metrics, &NodeMetrics {
            visits: 100,
            W: 0.0,
            children_visits: vec!(
                (CountingAction::Increment, 33),
                (CountingAction::Decrement, 27),
                (CountingAction::Stay, 39)
            )
        });
    }

    #[tokio::test]
    async fn test_mcts_works_with_single_node() {
        let game_state = CountingGameState::initial();
        let actions = 0;
        let game_engine = CountingGameEngine::new();
        let analyzer = CountingAnalyzer::new();

        let mut mcts = MCTS::new(game_state, actions, &game_engine, &analyzer, MCTSOptions::new(
            None,
            0.0,
            0.0,
            |_,_,_,_| 1.0,
            |_,_| 0.0,
            0.0,
            1
        ));

        mcts.search(1).await.unwrap();

        let metrics = mcts.get_root_node_metrics().await.unwrap();

        assert_metrics(&metrics, &NodeMetrics {
            visits: 1,
            W: 0.0,
            children_visits: vec!(
                (CountingAction::Increment, 0),
                (CountingAction::Decrement, 0),
                (CountingAction::Stay, 0)
            )
        });
    }

    #[tokio::test]
    async fn test_mcts_works_with_two_nodes() {
        let game_state = CountingGameState::initial();
        let actions = 0;
        let game_engine = CountingGameEngine::new();
        let analyzer = CountingAnalyzer::new();

        let mut mcts = MCTS::new(game_state, actions, &game_engine, &analyzer, MCTSOptions::new(
            None,
            0.0,
            0.0,
            |_,_,_,_| 1.0,
            |_,_| 0.0,
            0.0,
            1
        ));

        mcts.search(2).await.unwrap();

        let metrics = mcts.get_root_node_metrics().await.unwrap();

        assert_metrics(&metrics, &NodeMetrics {
            visits: 2,
            W: 0.0,
            children_visits: vec!(
                (CountingAction::Increment, 0),
                (CountingAction::Decrement, 0),
                (CountingAction::Stay, 1)
            )
        });
    }

    #[tokio::test]
    async fn test_mcts_works_from_provided_non_initial_game_state() {
        let game_state = CountingGameState::from_starting_count(true, 95);
        let actions = 0;
        let game_engine = CountingGameEngine::new();
        let analyzer = CountingAnalyzer::new();

        let mut mcts = MCTS::new(game_state, actions, &game_engine, &analyzer, MCTSOptions::new(
            None,
            0.0,
            0.0,
            |_,_,_,_| 0.1,
            |_,_| 0.0,
            0.0,
            1
        ));

        mcts.search(8000).await.unwrap();

        let metrics = mcts.get_root_node_metrics().await.unwrap();

        assert_metrics(&metrics, &NodeMetrics {
            visits: 8000,
            W: 0.0,
            children_visits: vec!(
                (CountingAction::Increment, 6470),
                (CountingAction::Decrement, 712),
                (CountingAction::Stay, 817)
            )
        });
    }

    #[tokio::test]
    async fn test_mcts_correctly_handles_terminal_nodes_1() {
        let game_state = CountingGameState::from_starting_count(false, 99);
        let actions = 0;
        let game_engine = CountingGameEngine::new();
        let analyzer = CountingAnalyzer::new();

        let mut mcts = MCTS::new(game_state, actions, &game_engine, &analyzer, MCTSOptions::new(
            None,
            0.0,
            0.0,
            |_,_,_,_| 1.0,
            |_,_| 0.0,
            0.0,
            1
        ));

        mcts.search(800).await.unwrap();

        let metrics = mcts.get_root_node_metrics().await.unwrap();

        assert_metrics(&metrics, &NodeMetrics {
            visits: 800,
            W: 0.0,
            children_visits: vec!(
                (CountingAction::Increment, 182),
                (CountingAction::Decrement, 312),
                (CountingAction::Stay, 305)
            )
        });
    }

    #[tokio::test]
    async fn test_mcts_correctly_handles_terminal_nodes_2() {
        let game_state = CountingGameState::from_starting_count(true, 98);
        let actions = 0;
        let game_engine = CountingGameEngine::new();
        let analyzer = CountingAnalyzer::new();

        let mut mcts = MCTS::new(game_state, actions, &game_engine, &analyzer, MCTSOptions::new(
            None,
            0.0,
            0.0,
            |_,_,_,_| 1.0,
            |_,_| 0.0,
            0.0,
            1
        ));

        mcts.search(800).await.unwrap();

        let metrics = mcts.get_root_node_metrics().await.unwrap();

        assert_metrics(&metrics, &NodeMetrics {
            visits: 800,
            W: 0.0,
            children_visits: vec!(
                (CountingAction::Increment, 316),
                (CountingAction::Decrement, 178),
                (CountingAction::Stay, 305)
            )
        });
    }

    #[tokio::test]
    async fn test_mcts_clear_nodes_results_in_same_outcome() {
        let game_state = CountingGameState::initial();
        let actions = 0;
        let game_engine = CountingGameEngine::new();
        let analyzer = CountingAnalyzer::new();
        let search_num_visits = 800;

        let mut non_clear_mcts = MCTS::new(game_state.clone(), actions.clone(), &game_engine, &analyzer, MCTSOptions::new(
            None,
            0.0,
            0.0,
            |_,_,_,_| 1.0,
            |_,_| 0.0,
            0.0,
            1
        ));

        non_clear_mcts.search(search_num_visits).await.unwrap();
        let action = non_clear_mcts.select_action().await.unwrap();
        non_clear_mcts.advance_to_action_retain(action).await.unwrap();
        non_clear_mcts.search(search_num_visits).await.unwrap();

        let non_clear_metrics = non_clear_mcts.get_root_node_metrics().await.unwrap();

        let mut clear_mcts = MCTS::new(game_state.clone(), actions.clone(), &game_engine, &analyzer, MCTSOptions::new(
            None,
            0.0,
            0.0,
            |_,_,_,_| 1.0,
            |_,_| 0.0,
            0.0,
            1
        ));

        clear_mcts.search(search_num_visits).await.unwrap();
        let action = clear_mcts.select_action().await.unwrap();
        clear_mcts.advance_to_action(action.clone()).await.unwrap();
        clear_mcts.search(search_num_visits).await.unwrap();

        let clear_metrics = clear_mcts.get_root_node_metrics().await.unwrap();

        let mut initial_mcts = MCTS::new(game_state, actions, &game_engine, &analyzer, MCTSOptions::new(
            None,
            0.0,
            0.0,
            |_,_,_,_| 1.0,
            |_,_| 0.0,
            0.0,
            1
        ));

        initial_mcts.advance_to_action(action).await.unwrap();
        initial_mcts.search(search_num_visits).await.unwrap();

        let initial_metrics = initial_mcts.get_root_node_metrics().await.unwrap();

        assert_metrics(&initial_metrics, &clear_metrics);
        assert_metrics(&non_clear_metrics, &clear_metrics);
    }
}
