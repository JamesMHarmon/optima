use std::cell::RefCell;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::{Arc,atomic::{AtomicBool,Ordering}};
use std::time::Duration;
use std::thread;
use futures::stream::{FuturesOrdered,StreamExt};
use generational_arena::{Arena,Index};
use rand::{thread_rng,Rng};
use rand::prelude::Distribution;
use rand::distributions::WeightedIndex;
use rand_distr::Dirichlet;
use failure::{Error,format_err};
use log::warn;

use engine::game_state::GameState;
use engine::value::Value;
use engine::engine::{GameEngine};
use model::analytics::{ActionWithPolicy,GameAnalyzer};
use model::node_metrics::{NodeMetrics};
use common::wait_for::WaitFor;
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
    logit_q: bool,
    cpuct: C,
    temperature: T,
    temperature_visit_offset: f32,
    moves_left_threshold: f32,
    moves_left_scale: f32,
    moves_left_factor: f32,
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
        logit_q: bool,
        cpuct: C,
        temperature: T,
        temperature_visit_offset: f32,
        moves_left_threshold: f32,
        moves_left_scale: f32,
        moves_left_factor: f32,
        parallelism: usize
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
    moves_left_score: f32,
    game_state: S,
    num_actions: usize,
    children: Vec<MCTSChildNode<A>>
}

#[derive(Debug)]
enum MCTSNodeState {
    Unexpanded,
    Expanding(Rc<WaitFor>),
    Expanded(Index)
}

#[allow(non_snake_case)]
#[derive(Debug)]
struct MCTSChildNode<A> {
    action: A,
    W: f32,
    visits: usize,
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

    pub async fn search_time(&mut self, duration: Duration) -> Result<usize, Error> {
        let alive = Arc::new(AtomicBool::new(true));

        let alive_clone = alive.clone();
        thread::spawn(move || {
            thread::sleep(duration);
            alive_clone.store(false, Ordering::SeqCst);
        });

        self.search(|_| alive.load(Ordering::SeqCst)).await
    }

    pub async fn search_visits(&mut self, visits: usize) -> Result<usize, Error> {
        let mut searches = 0;

        self.search(|initial_visits| {
            let prevsearches = searches;

            searches += 1;

            initial_visits + prevsearches < visits
        }).await
    }

    pub async fn play(&mut self, alive: &mut bool) -> Result<usize, Error> {
        self.search(|_| *alive).await
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

    pub async fn apply_noise_at_root(&mut self) {
        let root_node_index = self.get_or_create_root_node().await;

        if let Some(dirichlet) = &self.options.dirichlet { 
            let mut arena_borrow_mut = self.arena.borrow_mut();
            let root_node = &mut arena_borrow_mut[root_node_index];

            Self::apply_dirichlet_noise_to_node(root_node, dirichlet);
        }
    }

    pub async fn get_root_node_metrics(&mut self) -> Result<NodeMetrics<A>, Error> {
        let root_index = self.root.ok_or(format_err!("No root node found!"))?;
        let root = &self.arena.borrow()[root_index];

        Ok(NodeMetrics {
            visits: root.get_node_visits(),
            children: root.children.iter().map(|n| (
                n.action.clone(),
                if n.visits == 0 { 0.0 } else { n.W / n.visits as f32 },
                n.visits
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

    pub async fn search<F: FnMut(usize) -> bool>(&mut self, alive: F) -> Result<usize, Error> {
        let root_node_index = self.get_or_create_root_node().await;

        let game_engine = &self.game_engine;
        let options = &self.options;
        let arena_cell = &self.arena;
        let mut max_depth: usize = 0;
        let mut alive_flag = true;
        let mut alive = alive;

        let arena_borrow = arena_cell.borrow();
        let initial_visits = arena_borrow[root_node_index].get_node_visits();
        drop(arena_borrow);
        
        let analyzer = &mut self.analyzer;
        let mut searches = FuturesOrdered::new();

        for _ in 0..self.options.parallelism {
            if alive_flag && alive(initial_visits) {
                let future = recurse_path_and_expand::<S,A,E,M,C,T,V>(root_node_index, arena_cell, game_engine, analyzer, options);
                searches.push(future);
            } else {
                alive_flag = false;
            }
        }

        while let Some(search_depth) = searches.next().await {
            if alive_flag && alive(initial_visits) {
                let future = recurse_path_and_expand::<S,A,E,M,C,T,V>(root_node_index, arena_cell, game_engine, analyzer, options);
                searches.push(future);
            } else {
                alive_flag = false;
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

        let metrics = Self::get_PUCT_for_nodes(
            &root,
            &*arena_borrow,
            is_root,
            &self.options
        );

        let mut children: Vec<_> = root.children.iter().zip(metrics).map(|(n, m)| (
            n.action.clone(),
            m
        )).collect();

        children.sort_by(|(_, x_puct), (_, y_puct)| y_puct.cmp(&x_puct));

        Ok(NodeDetails {
            visits: root.get_node_visits(),
            children
        })
    }

    async fn advance_to_action_clearable(&mut self, action: A, clear: bool) -> Result<(), Error> {
        let root_index = self.get_or_create_root_node().await;

        let game_engine = &self.game_engine;

        let arena_borrow_mut = &mut *self.arena.borrow_mut();
        let root_node = arena_borrow_mut.remove(root_index).ok_or(format_err!("Root node should exist in arena."))?;
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
                Self::clear_node_children(node_index, arena_borrow_mut);
            }

            node_index
        } else {
            let prior_num_actions = root_node.num_actions;
            let new_game_state = game_engine.take_action(&root_node.game_state, &action);
            let analyzer = &mut self.analyzer;
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

    fn clear_node_children(node_index: Index, arena: &mut Arena<MCTSNode<S,A,V>>) {
        let node = &mut arena[node_index];

        for child in node.children.iter_mut() {
            child.visits = 0;
            child.W = 0.0;
        }

        let child_indexes: Vec<_> = node.children.iter().filter_map(|c| c.state.get_index()).collect();
        for child_index in child_indexes {
            Self::clear_node_children(child_index, arena);
        }
    }

    fn split_node_children_by_action<'b>(current_root: &'b MCTSNode<S,A,V>, action: &A) -> Result<(&'b MCTSNodeState, Vec<&'b MCTSNodeState>), Error> {
        let matching_action = current_root.get_child_of_action(action).ok_or(format_err!("No matching Action"))?;
        let other_actions: Vec<_> = current_root.children.iter().filter(|n| n.action != *action).map(|n| &n.state).collect();

        Ok((&matching_action.state, other_actions))
    }

    fn select_path(node_index: Index, arena: &Arena<MCTSNode<S,A,V>>, is_root: bool, options: &MCTSOptions<S,C,T>) -> Result<usize, Error> {
        let node = &arena[node_index];
        let children = &node.children;
        let fpu = if is_root { options.fpu_root } else { options.fpu };
        let Nsb = node.get_node_visits();
        let root_Nsb = (Nsb as f32).sqrt();
        let cpuct = &options.cpuct;
        let cpuct = cpuct(&node.game_state, node.num_actions, Nsb, is_root);
        let moves_left_baseline = get_moves_left_baseline(children, arena, options.moves_left_threshold);

        let mut best_child_index = 0;
        let mut best_puct = std::f32::MIN;

        for (i, child) in children.iter().enumerate() {
            let node = child.state.get_index().map(|i| &arena[i]);
            let W = child.W;
            let Nsa = child.visits;
            let Psa = child.policy_score;
            let Usa = cpuct * Psa * root_Nsb / (1 + Nsa) as f32;
            let Qsa = if Nsa == 0 { fpu } else { W / Nsa as f32 };
            let logitQ = if options.logit_q { logit(Qsa) } else { Qsa };
            let Msa = Self::get_Msa(node, moves_left_baseline, options);

            let PUCT = Msa + logitQ + Usa;

            if PUCT > best_puct {
                best_puct = PUCT;
                best_child_index = i;
            }
        }

        Ok(best_child_index)
    }

    fn select_action_using_temperature(action_visits: &[(&A, usize)], temp: f32, temperature_visit_offset: f32) -> Result<usize, Error> {
        let normalized_visits = action_visits.iter().map(|(_, visits)| (*visits as f32 + temperature_visit_offset).max(0.0).powf(1.0 / temp));

        let weighted_index = WeightedIndex::new(normalized_visits);

        let chosen_idx = match weighted_index {
            Err(_) => {
                warn!("Invalid puct scores. Most likely all are 0. Move will be randomly selected.");
                warn!("{:?}", action_visits);
                thread_rng().gen_range(0, action_visits.len())
            },
            Ok(weighted_index) => weighted_index.sample(&mut thread_rng())
        };

        Ok(chosen_idx)
    }

    fn get_PUCT_for_nodes(node: &MCTSNode<S,A,V>, arena: &Arena<MCTSNode<S,A,V>>, is_root: bool, options: &MCTSOptions<S,C,T>) -> Vec<PUCT>
    {
        let children = &node.children;
        let fpu = if is_root { options.fpu_root } else { options.fpu };
        let Nsb = node.get_node_visits();
        let root_Nsb = (Nsb as f32).sqrt();
        let cpuct = &options.cpuct;
        let cpuct = cpuct(&node.game_state, node.num_actions, Nsb, is_root);
        let moves_left_baseline = get_moves_left_baseline(&children, arena, options.moves_left_threshold);
        
        let mut pucts = Vec::with_capacity(children.len());

        for child in children {
            let node = child.state.get_index().map(|i| &arena[i]);
            let W = child.W;
            let Nsa = child.visits;
            let Psa = child.policy_score;
            let Usa = cpuct * Psa * root_Nsb / (1 + Nsa) as f32;
            let Qsa = if Nsa == 0 { fpu } else { W / Nsa as f32 };
            let logitQ = if options.logit_q { logit(Qsa) } else { Qsa };
            let moves_left = node.map_or(0.0, |n| n.moves_left_score);
            let Msa = Self::get_Msa(node, moves_left_baseline, options);

            let PUCT = logitQ + Usa;
            pucts.push(PUCT { Psa, Nsa, Msa, cpuct, Usa, Qsa, logitQ, moves_left, PUCT });
        }

        pucts
    }

    async fn expand_leaf(game_state: S, prior_num_actions: usize, analyzer: &M) -> MCTSNode<S,A,V> {
        let num_actions = prior_num_actions + 1;
        MCTS::<S,A,E,M,C,T,V>::analyse_and_create_node(game_state, num_actions, analyzer).await
    }

    async fn get_or_create_root_node(&mut self) -> Index {
        let root = &mut self.root;

        if let Some(root_node_index) = root.as_ref() {
            return *root_node_index;
        }

        let analyzer = &mut self.analyzer;
        let starting_num_actions = &mut self.starting_num_actions;
        let starting_game_state = &mut self.starting_game_state;

        let starting_game_state = starting_game_state.take().expect("Tried to use the same starting game state twice");
        let starting_num_actions = starting_num_actions.take().expect("Tried to use the same starting actions twice");

        let root_node = MCTS::<S,A,E,M,C,T,V>::analyse_and_create_node(
            starting_game_state,
            starting_num_actions,
            analyzer
        ).await;

        let root_node_index = self.arena.borrow_mut().insert(root_node);

        root.replace(root_node_index);

        root_node_index
    }

    async fn analyse_and_create_node(game_state: S, actions: usize, analyzer: &M) -> MCTSNode<S,A,V> {
        let analysis_result = analyzer.get_state_analysis(&game_state).await;

        MCTSNode::new(game_state, actions, analysis_result.value_score, analysis_result.policy_scores, analysis_result.moves_left)
    }

    fn apply_dirichlet_noise_to_node(node: &mut MCTSNode<S,A,V>, dirichlet: &DirichletOptions) {
        let policy_scores: Vec<f32> = node.children.iter().map(|child_node| {
            child_node.policy_score
        }).collect();

        let noisy_policy_scores = generate_noise(policy_scores, dirichlet);

        for (child, policy_score) in node.children.iter_mut().zip(noisy_policy_scores.into_iter()) {
            child.policy_score = policy_score;
        }
    }

    fn get_Msa(node: Option<&MCTSNode<S,A,V>>, moves_left_baseline: Option<f32>, options: &MCTSOptions<S,C,T>) -> f32
    {
        moves_left_baseline.and_then(|moves_left_baseline| node.map(|node| {
            let moves_left_scale = options.moves_left_scale;
            let moves_left_clamped = (moves_left_baseline - node.moves_left_score).min(moves_left_scale).max(-moves_left_scale);
            let moves_left_scaled = moves_left_clamped / moves_left_scale;
            moves_left_scaled * options.moves_left_factor
        }))
        .unwrap_or(0.0)
    }
}

fn generate_noise(policy_scores: Vec<f32>, dirichlet: &DirichletOptions) -> Vec<f32>
{
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

    dirichlet_noise.into_iter().zip(policy_scores).map(|(noise, policy_score)|
        (1.0 - e) * policy_score + e * noise
    ).collect()
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

#[allow(non_snake_case)]
fn get_moves_left_baseline<S,A,V>(nodes: &[MCTSChildNode<A>], arena: &Arena<MCTSNode<S,A,V>>, moves_left_threshold: f32) -> Option<f32> {
    nodes.iter()
        .max_by_key(|n| n.visits)
        .and_then(|best_child_node| best_child_node.state.get_index().map(|index| (best_child_node, index)))
        .and_then(|(best_child_node, best_node_index)| {
            let best_node = &arena[best_node_index];
            let Qsa = best_child_node.W / best_child_node.visits as f32;
            if Qsa >= moves_left_threshold {
                Some(best_node.moves_left_score)
            } else {
                None
            }
        })
}

#[allow(non_snake_case)]
impl<S,A,V> MCTSNode<S,A,V> {
    pub fn new(game_state: S, num_actions: usize, value_score: V, policy_scores: Vec<ActionWithPolicy<A>>, moves_left_score: f32) -> Self {
        MCTSNode {
            value_score,
            moves_left_score,
            game_state,
            num_actions,
            children: policy_scores.into_iter().map(|action_with_policy| {
                MCTSChildNode {
                    visits: 0,
                    W: 0.0,
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
    options: &MCTSOptions<S,C,T>
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
    let mut update_W_stack: Vec<(Index, usize)> = vec!();
    let mut latest_index = root_index;

    'outer: loop {
        depth += 1;
        let mut arena_borrow_mut = arena.borrow_mut();
        let node = &mut arena_borrow_mut[latest_index];

        // If the node is a terminal node.
        let children = &node.children;
        if children.len() == 0 {
            update_Ws(&update_W_stack, latest_index, &mut arena_borrow_mut, game_engine);
            break 'outer;
        }

        let is_root = depth == 1;

        let selected_child_node_children_index = MCTS::<S,A,E,M,C,T,V>::select_path(
            latest_index,
            &mut *arena_borrow_mut,
            is_root,
            options
        )?;

        update_W_stack.push((latest_index, selected_child_node_children_index));

        let selected_child_node = &mut arena_borrow_mut[latest_index].children[selected_child_node_children_index];
        let prev_visits = selected_child_node.visits;
        selected_child_node.visits += 1;

        if let MCTSNodeState::Expanded(selected_child_node_index) = selected_child_node.state {
            // If the node exists but visits was 0, then this node was cleared but the analysis was saved. Treat it as such by keeping the values.
            if prev_visits == 0 {
                update_Ws(&update_W_stack, selected_child_node_index, &mut arena_borrow_mut, game_engine);
                break 'outer;
            }

            // Continue with the next iteration of the loop since we found an already expanded child node.
            latest_index = selected_child_node_index;
            continue 'outer;
        }

        let selected_action = selected_child_node.action.clone();
        drop(arena_borrow_mut);

        'inner: loop {
            let arena_borrow = arena.borrow();
            let prior_node = &arena_borrow[latest_index];
            let selected_child_node = prior_node.get_child_of_action(&selected_action).unwrap();
            let selected_child_node_state = &selected_child_node.state;
            if let MCTSNodeState::Expanded(selected_child_node_index) = selected_child_node.state {
                latest_index = selected_child_node_index;
                continue 'outer;
            }

            // If the node is currently expanding. Wait for the expansion to be completed.
            if let MCTSNodeState::Expanding(expanding_lock) = selected_child_node_state {
                let expanding_lock = expanding_lock.clone();
                drop(arena_borrow);

                expanding_lock.wait().await;

                continue 'inner;
            }

            // It is not expanded or expanding so let's expand it.
            drop(arena_borrow);
            let mut arena_borrow_mut = arena.borrow_mut();
            let prior_node = &mut arena_borrow_mut[latest_index];
            let selected_child_node = prior_node.get_child_of_action_mut(&selected_action).unwrap();
            let selected_child_node_state = &mut selected_child_node.state;

            // Double check that the node has not changed now that the lock has been reacquired as a write.
            if let MCTSNodeState::Unexpanded = selected_child_node_state {
                // Immediately replace the state with an indication that we are expanding.RwLock
                let expanding_lock = Rc::new(WaitFor::new());
                std::mem::replace(selected_child_node_state, MCTSNodeState::Expanding(expanding_lock.clone()));

                let new_game_state = game_engine.take_action(&prior_node.game_state, &selected_action);
                let prior_num_actions = prior_node.num_actions;

                drop(arena_borrow_mut);

                let expanded_node = MCTS::<S,A,E,M,C,T,V>::expand_leaf(
                    new_game_state,
                    prior_num_actions,
                    analyzer
                ).await;

                let arena_borrow_mut = &mut arena.borrow_mut();

                let expanded_node_index = update_child_with_expanded_node(latest_index, expanded_node, &selected_action, arena_borrow_mut);

                update_Ws(&update_W_stack, expanded_node_index, arena_borrow_mut, game_engine);

                expanding_lock.wake();

                break 'outer;
            }
        }
    }

    Ok(depth)
}

fn update_child_with_expanded_node<S,A,V>(parent_node: Index, expanded_node: MCTSNode<S,A,V>, action: &A, arena: &mut Arena<MCTSNode<S,A,V>>) -> Index
where
    A: Eq
{
    let expanded_node_index = arena.insert(expanded_node);
    let prior_node = &mut arena[parent_node];
    let selected_child_node = prior_node.get_child_of_action_mut(action).unwrap();
    let selected_child_node_state = &mut selected_child_node.state;
    std::mem::replace(selected_child_node_state, MCTSNodeState::Expanded(expanded_node_index));

    expanded_node_index
}

#[allow(non_snake_case)]
fn update_Ws<S,A,V,E>(node_indexes: &[(Index, usize)], value_node_index: Index, arena: &mut Arena<MCTSNode<S,A,V>>, game_engine: &E)
where
    V: Value,
    E: GameEngine<State=S,Action=A,Value=V>
{
    let value_node = &arena[value_node_index];
    let value_score = &value_node.value_score.clone();

    for (node_index, children_index) in node_indexes {
        let node = &mut arena[*node_index];
        // Update value of W from the parent node's perspective.
        // This is because the parent chooses which child node to select, and as such will want the one with the
        // highest V from it's perspective. A node never cares what its value (W or Q) is from its own perspective.
        let player_to_move = game_engine.get_player_to_move(&node.game_state);
        let score = value_score.get_value_for_player(player_to_move);
        node.children[*children_index].W += score;
    }
}

impl<S,A,V> MCTSNode<S,A,V>
where
    A: Eq
{
    fn get_node_visits(&self) -> usize {
        self.children.iter().map(|c| c.visits).sum::<usize>() + 1
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
    const ERROR_DIFF_W: f32 = 0.01;

    fn assert_metrics(left: &NodeMetrics<CountingAction>, right: &NodeMetrics<CountingAction>) {
        assert_eq!(left.visits, right.visits);
        assert_eq!(left.children.len(), right.children.len());

        for ((l_a, l_w, l_visits), (r_a, r_w, r_visits)) in left.children.iter().zip(right.children.iter()) {
            assert_eq!(l_a, r_a);
            assert_approx_eq!(l_w, r_w, ERROR_DIFF_W);
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
            true,
            |_,_,_,_| 3.0,
            |_,_| 0.0,
            0.0,
            1.0,
            10.0,
            0.05,
            1
        ));

        let mut mcts2 = MCTS::new(game_state, actions, &game_engine, &analyzer, MCTSOptions::new(
            None,
            0.0,
            0.0,
            true,
            |_,_,_,_| 3.0,
            |_,_| 0.0,
            0.0,
            1.0,
            10.0,
            0.05,
            1
        ));

        mcts.search_visits(800).await.unwrap();
        mcts2.search_visits(800).await.unwrap();

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
            true,
            |_,_,_,_| 1.0,
            |_,_| 0.0,
            0.0,
            1.0,
            10.0,
            0.05,
            1
        ));

        mcts.search_visits(800).await.unwrap();
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
            true,
            |_,_,_,_| 1.0,
            |_,_| 0.0,
            0.0,
            1.0,
            10.0,
            0.05,
            1
        ));

        mcts.advance_to_action(CountingAction::Increment).await.unwrap();

        mcts.search_visits(800).await.unwrap();
        let action = mcts.select_action().await.unwrap();

        assert_eq!(action, CountingAction::Decrement);
    }

    #[tokio::test]
    async fn test_mcts_should_overcome_policy_through_value() {
        let game_state = CountingGameState::initial();
        let actions = 0;
        let game_engine = CountingGameEngine::new();
        let analyzer = CountingAnalyzer::new();

        let mut mcts = MCTS::new(game_state, actions, &game_engine, &analyzer, MCTSOptions::new(
            None,
            0.0,
            0.0,
            true,
            |_,_,_,_| 2.5,
            |_,_| 0.0,
            0.0,
            1.0,
            10.0,
            0.05,
            1
        ));

        mcts.advance_to_action(CountingAction::Increment).await.unwrap();

        mcts.search_visits(800).await.unwrap();
        let details = mcts.get_root_node_details().await.unwrap();
        let (action, _) = details.children.first().unwrap();

        assert_eq!(*action, CountingAction::Stay);

        mcts.search_visits(8000).await.unwrap();
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
            true,
            |_,_,_,_| 3.0,
            |_,_| 0.0,
            0.0,
            1.0,
            10.0,
            0.05,
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
            true,
            |_,_,_,_| 1.0,
            |_,_| 0.0,
            0.0,
            1.0,
            10.0,
            0.05,
            1
        ));

        mcts.search_visits(800).await.unwrap();

        let metrics = mcts.get_root_node_metrics().await.unwrap();

        assert_metrics(&metrics, &NodeMetrics {
            visits: 800,
            children: vec!(
                (CountingAction::Increment, 0.509, 312),
                (CountingAction::Decrement, 0.49, 182),
                (CountingAction::Stay, 0.5, 304)
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
            true,
            |_,_,_,_| 3.0,
            |_,_| 0.0,
            0.0,
            1.0,
            10.0,
            0.05,
            1
        ));

        mcts.search_visits(100).await.unwrap();

        let metrics = mcts.get_root_node_metrics().await.unwrap();

        assert_metrics(&metrics, &NodeMetrics {
            visits: 100,
            children: vec!(
                (CountingAction::Increment, 0.51, 31),
                (CountingAction::Decrement, 0.49, 29),
                (CountingAction::Stay, 0.5, 40)
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
            true,
            |_,_,_,_| 3.0,
            |_,_| 0.0,
            0.0,
            1.0,
            10.0,
            0.05,
            1
        ));

        mcts.search_visits(1).await.unwrap();

        let metrics = mcts.get_root_node_metrics().await.unwrap();

        assert_metrics(&metrics, &NodeMetrics {
            visits: 1,
            children: vec!(
                (CountingAction::Increment, 0.0, 0),
                (CountingAction::Decrement, 0.0, 0),
                (CountingAction::Stay, 0.0, 0)
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
            true,
            |_,_,_,_| 3.0,
            |_,_| 0.0,
            0.0,
            1.0,
            10.0,
            0.05,
            1
        ));

        mcts.search_visits(2).await.unwrap();

        let metrics = mcts.get_root_node_metrics().await.unwrap();

        assert_metrics(&metrics, &NodeMetrics {
            visits: 2,
            children: vec!(
                (CountingAction::Increment, 0.0, 0),
                (CountingAction::Decrement, 0.0, 0),
                (CountingAction::Stay, 0.5, 1)
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
            true,
            |_,_,_,_| 3.0,
            |_,_| 0.0,
            0.0,
            1.0,
            10.0,
            0.05,
            1
        ));

        mcts.search_visits(8000).await.unwrap();

        let metrics = mcts.get_root_node_metrics().await.unwrap();

        assert_metrics(&metrics, &NodeMetrics {
            visits: 8000,
            children: vec!(
                (CountingAction::Increment, 0.956, 5374),
                (CountingAction::Decrement, 0.938, 798),
                (CountingAction::Stay, 0.948, 1827)
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
            true,
            |_,_,_,_| 3.0,
            |_,_| 0.0,
            0.0,
            1.0,
            10.0,
            0.05,
            1
        ));

        mcts.search_visits(800).await.unwrap();

        let metrics = mcts.get_root_node_metrics().await.unwrap();

        assert_metrics(&metrics, &NodeMetrics {
            visits: 800,
            children: vec!(
                (CountingAction::Increment, 0.0, 8),
                (CountingAction::Decrement, 0.02, 701),
                (CountingAction::Stay, 0.005, 91)
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
            true,
            |_,_,_,_| 3.0,
            |_,_| 0.0,
            0.0,
            1.0,
            10.0,
            0.05,
            1
        ));

        mcts.search_visits(800).await.unwrap();

        let metrics = mcts.get_root_node_metrics().await.unwrap();

        assert_metrics(&metrics, &NodeMetrics {
            visits: 800,
            children: vec!(
                (CountingAction::Increment, 0.982, 400),
                (CountingAction::Decrement, 0.968, 122),
                (CountingAction::Stay, 0.977, 278)
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
            true,
            |_,_,_,_| 3.0,
            |_,_| 0.0,
            0.0,
            1.0,
            10.0,
            0.05,
            1
        ));

        non_clear_mcts.search_visits(search_num_visits).await.unwrap();
        let action = non_clear_mcts.select_action().await.unwrap();
        non_clear_mcts.advance_to_action_retain(action).await.unwrap();
        non_clear_mcts.search_visits(search_num_visits).await.unwrap();

        let non_clear_metrics = non_clear_mcts.get_root_node_metrics().await.unwrap();

        let mut clear_mcts = MCTS::new(game_state.clone(), actions.clone(), &game_engine, &analyzer, MCTSOptions::new(
            None,
            0.0,
            0.0,
            true,
            |_,_,_,_| 3.0,
            |_,_| 0.0,
            0.0,
            1.0,
            10.0,
            0.05,
            1
        ));

        clear_mcts.search_visits(search_num_visits).await.unwrap();
        let action = clear_mcts.select_action().await.unwrap();
        clear_mcts.advance_to_action(action.clone()).await.unwrap();
        clear_mcts.search_visits(search_num_visits).await.unwrap();

        let clear_metrics = clear_mcts.get_root_node_metrics().await.unwrap();

        let mut initial_mcts = MCTS::new(game_state, actions, &game_engine, &analyzer, MCTSOptions::new(
            None,
            0.0,
            0.0,
            true,
            |_,_,_,_| 3.0,
            |_,_| 0.0,
            0.0,
            1.0,
            10.0,
            0.05,
            1
        ));

        initial_mcts.advance_to_action(action).await.unwrap();
        initial_mcts.search_visits(search_num_visits).await.unwrap();

        let initial_metrics = initial_mcts.get_root_node_metrics().await.unwrap();

        assert_metrics(&initial_metrics, &clear_metrics);
        assert_metrics(&non_clear_metrics, &clear_metrics);
    }
}
