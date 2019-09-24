use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::{self,Arc,Mutex};
use futures::stream::{FuturesUnordered,StreamExt};
use async_std::sync::RwLock;
use generational_arena::{Arena,Index};
use sync::atomic::{AtomicUsize,Ordering};
use rand::{thread_rng,Rng};
use rand::prelude::Distribution;
use rand::distributions::WeightedIndex;
use rand_distr::Dirichlet;
use failure::{Error,format_err};

use engine::game_state::GameState;
use engine::engine::{GameEngine};
use model::analytics::{ActionWithPolicy,GameAnalyzer};
use model::node_metrics::{NodeMetrics};
use common::linked_list::{List};
use super::node_details::{PUCT,NodeDetails};

pub struct DirichletOptions {
    pub alpha: f32,
    pub epsilon: f32
}

pub struct MCTSOptions<S, A, C, T>
where
    S: GameState
{
    dirichlet: Option<DirichletOptions>,
    fpu: f32,
    fpu_root: f32,
    cpuct: C,
    temperature: T,
    _phantom_action: PhantomData<*const A>,
    _phantom_state: PhantomData<*const S>
}

impl<S, A, C, T> MCTSOptions<S, A, C, T>
where
    S: GameState,
    A: Clone + Eq + Debug,
    C: Fn(&S, &List<A>, &A, usize, bool) -> f32,
    T: Fn(&S, &List<A>) -> f32
{
    pub fn new(
        dirichlet: Option<DirichletOptions>,
        fpu: f32,
        fpu_root: f32,
        cpuct: C,
        temperature: T
    ) -> Self {
        MCTSOptions {
            dirichlet,
            fpu,
            fpu_root,
            cpuct,
            temperature,
            _phantom_action: PhantomData,
            _phantom_state: PhantomData
        }
    }
}

pub struct MCTS<'a, S, A, E, M, C, T>
where
    S: GameState,
    A: Clone + Eq + Debug,
    E: GameEngine,
    M: GameAnalyzer,
    C: Fn(&S, &List<A>, &A, usize, bool) -> f32,
    T: Fn(&S, &List<A>) -> f32
{
    options: MCTSOptions<S, A, C, T>,
    game_engine: &'a E,
    analytics: &'a M,
    starting_game_state: Option<S>,
    starting_actions: Option<List<A>>,
    root: Option<Index>,
    arena: RwLock<Arena<MCTSNode<S, A>>>
}

#[allow(non_snake_case)]
#[derive(Debug)]
struct MCTSNode<S, A> {
    visits: AtomicUsize,
    value_score: f32,
    W: Mutex<f32>,
    game_state: S,
    actions: List<A>,
    children: Vec<MCTSChildNode<A>>
}

#[derive(Debug)]
enum MCTSNodeState {
    Unexpanded,
    Expanding(Arc<RwLock<()>>),
    Expanded(Index)
}

#[derive(Debug)]
struct MCTSChildNode<A> {
    action: A,
    policy_score: f32,
    state: MCTSNodeState
}

struct NodePUCT<'a, A> {
    node: &'a MCTSChildNode<A>,
    score: f32
}

struct StateAnalysisValue {
    value_score: f32
}

#[allow(non_snake_case)]
impl<'a, S, A, E, M, C, T> MCTS<'a, S, A, E, M, C, T>
where
    S: GameState,
    A: Clone + Eq + Debug,
    E: 'a + GameEngine<State=S,Action=A>,
    M: 'a + GameAnalyzer<State=S,Action=A>,
    C: Fn(&S, &List<A>, &A, usize, bool) -> f32,
    T: Fn(&S, &List<A>) -> f32
{
    pub fn new(
        game_state: S,
        actions: List<A>,
        game_engine: &'a E,
        analytics: &'a M,
        options: MCTSOptions<S, A, C, T>
    ) -> Self {
        MCTS {
            options,
            game_engine,
            analytics,
            starting_game_state: Some(game_state),
            starting_actions: Some(actions),
            root: None,
            arena: RwLock::new(Arena::new())
        }
    }

    pub async fn search(&mut self, visits: usize) -> Result<usize, Error> {
        let game_engine = &self.game_engine;
        let fpu = self.options.fpu;
        let fpu_root = self.options.fpu_root;
        let cpuct = &self.options.cpuct;
        let dirichlet = &self.options.dirichlet;
        let analytics = &mut self.analytics;
        let root = &mut self.root;
        let starting_actions = &mut self.starting_actions;
        let starting_game_state = &mut self.starting_game_state;
        let arena = &self.arena;
        let root_node_index = MCTS::<S,A,E,M,C,T>::get_or_create_root_node(
            root,
            starting_game_state,
            starting_actions,
            analytics,
            arena,
            dirichlet
        ).await;

        let mut max_depth: usize = 0;

        let mut searches = FuturesUnordered::new();
        for _ in 0..visits {
            let future = recurse_path_and_expand::<S,A,E,M,C,T>(root_node_index, arena, game_engine, analytics, fpu, fpu_root, cpuct);

            searches.push(future);
        }

        while let Some(search_depth) = searches.next().await {
            max_depth = max_depth.max(search_depth?);
        }

        Ok(max_depth)
    }

    pub async fn select_action(&mut self) -> Result<A, Error> {
        if let Some(root_node_index) = &self.root {
            let root_node = &self.arena.read().await[*root_node_index];
            let temp = &self.options.temperature;
            let game_state = &root_node.game_state;
            let prior_actions = &root_node.actions;
            let temp = temp(game_state, prior_actions);
            let child_node_details = self.get_root_node_details().await?.children;

            let best_action = if temp == 0.0 {
                let (best_action, _) = child_node_details.first().ok_or_else(|| format_err!("No available actions"))?;
                best_action
            } else {
                let candidates: Vec<_> = child_node_details.iter().map(|(a, puct)| (a, puct.Nsa)).collect();
                let chosen_index = Self::select_path_using_temperature(&candidates, temp)?;
                candidates[chosen_index].0
            };

            return Ok(best_action.clone());
        }

        return Err(format_err!("Root node does not exist. Run search first."));
    }

    pub async fn advance_to_action(&mut self, action: A) -> Result<(), Error> {
        self.advance_to_action_clearable(action, true).await
    }

    pub async fn advance_to_action_retain(&mut self, action: A) -> Result<(), Error> {
        self.advance_to_action_clearable(action, false).await
    }

    pub async fn get_root_node_metrics(&mut self) -> Result<NodeMetrics<A>, Error> {
        let root_index = self.root.ok_or(format_err!("No root node found!"))?;
        let arena = self.arena.read().await;
        let root = &arena[root_index];
        let visits = root.visits.load(Ordering::SeqCst);
        let W_lock = root.W.lock().unwrap();
        let W = *W_lock;
        drop(W_lock);

        Ok(NodeMetrics {
            visits,
            W,
            children_visits: root.children.iter().map(|n| (
                n.action.clone(),
                if let MCTSNodeState::Expanded(node_index) = n.state {
                    let node = &arena[node_index];
                    node.visits.load(Ordering::SeqCst)
                } else { 0 }
            )).collect()
        })
    }

    pub async fn get_root_node_details(&self) -> Result<NodeDetails<A>, Error> {
        let arena = &self.arena;
        let root_index = self.root.as_ref().ok_or(format_err!("No root node found!"))?;
        let arena_read_lock = arena.read().await;
        let root = &arena_read_lock[*root_index];
        let root_visits = root.visits.load(Ordering::SeqCst);
        let children = &root.children;
        let options = &self.options;

        let metrics = Self::get_PUCT_for_nodes(
            children,
            arena,
            root_visits,
            &root.game_state,
            true,
            &root.actions,
            options.fpu,
            options.fpu_root,
            &options.cpuct
        ).await;

        let mut children: Vec<_> = children.iter().zip(metrics).map(|(n, m)| (
            n.action.clone(),
            m
        )).collect();

        children.sort_by(|(_, x_puct), (_, y_puct)| y_puct.cmp(&x_puct));

        let W = *root.W.lock().unwrap();

        Ok(NodeDetails {
            visits: root_visits,
            W,
            children
        })
    }

    async fn advance_to_action_clearable(&mut self, action: A, clear: bool) -> Result<(), Error> {
        let mut root = self.root.take();
        let analytics = &mut self.analytics;
        let starting_game_state = &mut self.starting_game_state;
        let starting_actions = &mut self.starting_actions;
        let game_engine = &self.game_engine;
        let dirichlet = &self.options.dirichlet;

        let arena = &self.arena;
        let root_index = MCTS::<S,A,E,M,C,T>::get_or_create_root_node(&mut root, starting_game_state, starting_actions, analytics, arena, dirichlet).await;

        let arena = &mut *arena.write().await;
        let root_node = arena.remove(root_index).expect("Root node should exist in arena.");
        let node = Self::get_index_of_action(&root_node, &action);

        if let Err(err) = node {
            // If there is an error, replace the root node back to it's original value.
            let index = arena.insert(root_node);
            self.root = Some(index);
            return Err(err);
        }

        let node = node.unwrap();

        let node = match node {
            MCTSNodeState::Expanded(node_index) => {
                if clear { Self::clear_node_visits(*node_index, arena); }
                // If the node was cleared then the visits may still be 0. This should be incremented to 1 if that is the case.
                // This condition can occur even if clear is false.
                let node = &mut arena[*node_index];
                node.visits.compare_and_swap(0, 1, Ordering::SeqCst);
                *node_index
            },
            _ => {
                let prior_actions = &root_node.actions;
                let (node, _) = MCTS::<S,A,E,M,C,T>::expand_leaf(&root_node.game_state, prior_actions, &action, game_engine, analytics).await;
                arena.insert(node)
            }
        };

        self.root.replace(node);

        Ok(())
    }

    fn clear_node_visits(node_index: Index, arena: &mut Arena<MCTSNode<S,A>>) {
        let node = &mut arena[node_index];
        node.visits.store(0, Ordering::SeqCst);
        *node.W.lock().unwrap() = node.value_score;

        let child_indexes: Vec<_> = node.children.iter().filter_map(|child| {
            if let MCTSNodeState::Expanded(child_node_index) = child.state { Some(child_node_index) } else { None }
        }).collect();

        for child_index in child_indexes {
            Self::clear_node_visits(child_index, arena);
        }
    }

    fn get_index_of_action<'b>(current_root: &'b MCTSNode<S, A>, action: &A) -> Result<&'b MCTSNodeState, Error> {
        let matching_action = current_root.children.iter().find(|n| n.action == *action).ok_or(format_err!("No matching Action"))?;

        Ok(&matching_action.state)
    }

    async fn select_path(nodes: &'a [MCTSChildNode<A>], arena: &'a RwLock<Arena<MCTSNode<S,A>>>, Nsb: usize, game_state: &S, is_root: bool, prior_actions: &List<A>, fpu: f32, fpu_root: f32, cpuct: &C) -> Result<&'a MCTSChildNode<A>, Error> {
        let mut pucts = Self::get_PUCT_for_nodes_mut(nodes, arena, Nsb, game_state, is_root, prior_actions, fpu, fpu_root, cpuct).await;

        let chosen_puct_idx = Self::get_max_PUCT_score_index(&pucts)?;

        Ok(pucts.swap_remove(chosen_puct_idx).node)
    }

    fn get_max_PUCT_score_index(pucts: &Vec<NodePUCT<A>>) -> Result<usize, Error> {
        let max_puct = pucts.iter().fold(std::f32::MIN, |acc, puct| f32::max(acc, puct.score));
        let mut max_nodes: Vec<usize> = pucts.into_iter().enumerate()
            .filter_map(|(i, puct)| if puct.score >= max_puct { Some(i) } else { None })
            .collect();
    
        match max_nodes.len() {
            0 => Err(format_err!("No candidate moves available")),
            1 => Ok(max_nodes.swap_remove(0)),
            len => Ok(max_nodes.swap_remove(thread_rng().gen_range(0, len)))
        }
    }

    fn select_path_using_temperature(action_visits: &[(&A, usize)], temp: f32) -> Result<usize, Error> {
        let normalized_visits = action_visits.iter().map(|(_, visits)| (*visits as f32).powf(1.0 / temp));

        let weighted_index = WeightedIndex::new(normalized_visits);

        let chosen_idx = match weighted_index {
            Err(_) => {
                println!("Invalid puct scores. Most likely all are 0. Move will be randomly selected.");
                thread_rng().gen_range(0, action_visits.len())
            },
            Ok(weighted_index) => weighted_index.sample(&mut thread_rng())
        };

        Ok(chosen_idx)
    }

    async fn get_PUCT_for_nodes_mut(nodes: &'a [MCTSChildNode<A>], arena: &'a RwLock<Arena<MCTSNode<S,A>>>, Nsb: usize, game_state: &S, is_root: bool, prior_actions: &List<A>, fpu: f32, fpu_root: f32, cpuct: &C) -> Vec<NodePUCT<'a, A>>
    {
        let pucts = Self::get_PUCT_for_nodes(nodes, arena, Nsb, game_state, is_root, prior_actions, fpu, fpu_root, cpuct).await;
        nodes.iter().zip(pucts).map(|(node, puct)| {
            NodePUCT::<A> { node, score: puct.PUCT }
        }).collect()
    }

    async fn get_PUCT_for_nodes(nodes: &[MCTSChildNode<A>], arena: &'a RwLock<Arena<MCTSNode<S,A>>>, Nsb: usize, game_state: &S, is_root: bool, prior_actions: &List<A>, fpu: f32, fpu_root: f32, cpuct: &C) -> Vec<PUCT>
    {
        let fpu = if is_root { fpu_root } else { fpu };
        let mut pucts = Vec::with_capacity(nodes.len());
        let arena = arena.read().await;

        for child in nodes {
            // If the child nodes visits is 0, then it has been cleared and should have it's cpuct be calculated as if it is a leaf.
            let (Nsa, W) = if let MCTSNodeState::Expanded(index) = child.state {
                let node = &arena[index];
                let visits = node.visits.load(Ordering::SeqCst);
                let W = *node.W.lock().unwrap();
                (visits, W)
            } else {
                (0, 0.0)
            };

            let Psa = child.policy_score;
            let cpuct = cpuct(game_state, prior_actions, &child.action, Nsb, is_root);
            let root_Nsb = (Nsb as f32).sqrt();
            let Usa = cpuct * Psa * root_Nsb / (1 + Nsa) as f32;

            // Reverse W here since the evaluation of each child node is that from the other player's perspective.
            let Qsa = if Nsa == 0 { fpu } else { 1.0 - W / Nsa as f32 };
            let PUCT = Qsa + Usa;
            pucts.push(PUCT { Psa, Nsa, cpuct, Usa, Qsa, PUCT });
        }

        pucts
    }

    async fn expand_leaf(prior_game_state: &'a S, prior_actions: &'a List<A>, action: &'a A, game_engine: &'a E, analytics: &'a M) -> (MCTSNode<S, A>, StateAnalysisValue) {
        let new_game_state = game_engine.take_action(prior_game_state, action);
        let new_actions = prior_actions.append(action.to_owned());
        MCTS::<S,A,E,M,C,T>::analyse_and_create_node(new_game_state, new_actions, analytics).await
    }

    async fn get_or_create_root_node(
        root: &mut Option<Index>,
        starting_game_state: &mut Option<S>,
        starting_actions: &mut Option<List<A>>,
        analytics: &M,
        arena: &RwLock<Arena<MCTSNode<S,A>>>,
        dirichlet: &Option<DirichletOptions>
    ) -> Index {
        if let Some(root_node_index) = root.as_ref() {
            return *root_node_index;
        }

        let starting_game_state = starting_game_state.take().expect("Tried to use the same starting game state twice");
        let starting_actions = starting_actions.take().expect("Tried to use the same starting actions twice");

        let (mut root_node, _) = MCTS::<S,A,E,M,C,T>::analyse_and_create_node(
            starting_game_state,
            starting_actions,
            analytics
        ).await;

        Self::apply_dirichlet_noise_to_node(&mut root_node, dirichlet);

        let root_node_index = arena.write().await.insert(root_node);

        root.replace(root_node_index);

        root_node_index
    }

    // Value range is [-1, 1] for the "get_state_analysis" method. However internally for the MCTS a range of
    // [0, 1] is used.
    async fn analyse_and_create_node(game_state: S, actions: List<A>, analytics: &'a M) -> (MCTSNode<S, A>, StateAnalysisValue) {
        let analysis_result = analytics.get_state_analysis(&game_state).await;

        let value_score = (analysis_result.value_score + 1.0) / 2.0;

        (
            MCTSNode::new(game_state, actions, value_score, analysis_result.policy_scores),
            StateAnalysisValue { value_score }
        )
    }

    fn apply_dirichlet_noise_to_node(node: &mut MCTSNode<S, A>, dirichlet: &Option<DirichletOptions>) {
        if let Some(dirichlet) = dirichlet {
            let policy_scores: Vec<f32> = node.children.iter().map(|child_node| {
                child_node.policy_score
            }).collect();

            let updated_policy_scores = MCTS::<S,A,E,M,C,T>::apply_dirichlet_noise(policy_scores, dirichlet);

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

impl<S, A> MCTSNode<S, A> {
    pub fn new(game_state: S, actions: List<A>, value_score: f32, policy_scores: Vec<ActionWithPolicy<A>>) -> Self {
        MCTSNode {
            visits: AtomicUsize::new(1),
            value_score,
            W: Mutex::new(value_score),
            game_state,
            actions,
            children: policy_scores.into_iter().map(|action_with_policy| {
                MCTSChildNode {
                    action: action_with_policy.action,
                    policy_score: action_with_policy.policy_score,
                    state: MCTSNodeState::Unexpanded
                }
            }).collect()
        }
    }
}

#[allow(non_snake_case)]
async fn recurse_path_and_expand<'a,S,A,E,M,C,T>(
    root_index: Index,
    arena: &'a RwLock<Arena<MCTSNode<S,A>>>,
    game_engine: &'a E,
    analytics: &'a M,
    fpu: f32,
    fpu_root: f32,
    cpuct: &'a C
) -> Result<usize, Error>
where
    S: GameState,
    A: Clone + Eq + Debug,
    E: GameEngine<State=S,Action=A>,
    M: GameAnalyzer<State=S,Action=A>,
    C: Fn(&S, &List<A>, &A, usize, bool) -> f32,
    T: Fn(&S, &List<A>) -> f32
{
    let mut depth = 0;
    let value_score: f32;
    let mut node_stack = vec!(root_index);

    loop {
        depth += 1;
        if let Some(latest_index) = node_stack.last() {
            let node_read_lock = arena.read().await;
            let node = &node_read_lock[*latest_index];
            let prev_visits = node.visits.fetch_add(1, Ordering::SeqCst);

            // @TODO: ADD back!!
            // Ws_to_update.push(&node.W);

            // If the node is a terminal node.
            let children = &node.children;
            if children.len() == 0 {
                value_score = node.value_score as f32;
                break;
            }

            let game_state = &node.game_state;
            let prior_actions = &node.actions;
            let is_root = depth == 1;
            let selected_child_node = MCTS::<S,A,E,M,C,T>::select_path(
                children,
                arena,
                prev_visits,
                game_state,
                is_root,
                prior_actions,
                fpu,
                fpu_root,
                cpuct
            ).await?;

            if let MCTSNodeState::Expanded(selected_child_node_index) = selected_child_node.state {
                let node = &node_read_lock[selected_child_node_index];
                // If the node exists but visits is 0, then this node was cleared but the analysis was saved. Treat it as such by keeping the values.
                if node.visits.compare_and_swap(0, 1, Ordering::SeqCst) == 0 {
                    // The compare_and_swap above will update visits to be 1.
                    // node.visits = 1;
                    // Flip the score in this case because we are going one node deeper and that viewpoint is from
                    // the next player and not the current node's player
                    value_score = 1.0 - node.value_score;
                    break;
                }

                // Continue with the next iteration of the loop since we found an already expanded child node.
                node_stack.push(selected_child_node_index);
                continue;
            }

            let selected_action = selected_child_node.action.clone();
            drop(node_read_lock);


            let mut node_write_lock = arena.write().await;
            let arena_write = &mut *node_write_lock;
            let prior_node = &mut arena_write[*latest_index];
            let selected_child_node = prior_node.children.iter_mut().find(|c| c.action == selected_action).unwrap();
            let selected_child_node_state = &mut selected_child_node.state;

            // Double check that the node has not changed now that the lock has been reacquired as a write.
            if let MCTSNodeState::Unexpanded = selected_child_node_state {
                // Immediately replace the state with an indication that we are expanding.RwLock
                let expanding_lock = std::sync::Arc::new(RwLock::new(()));
                let expanding_write_lock = expanding_lock.write().await;
                std::mem::replace(selected_child_node_state, MCTSNodeState::Expanding(expanding_lock.clone()));

                let action = selected_child_node.action.clone();
                let prior_game_state = prior_node.game_state.clone();
                let prior_actions = prior_node.actions.clone();

                drop(node_write_lock);

                let fut = MCTS::<S,A,E,M,C,T>::expand_leaf(
                    &prior_game_state,
                    &prior_actions,
                    &action,
                    game_engine,
                    analytics
                );

                let (expanded_node, state_analysis) = fut.await;

                let mut node_write_lock = arena.write().await;
                let arena_write = &mut *node_write_lock;
                let index = arena_write.insert(expanded_node);
                let prior_node = &mut arena_write[*latest_index];
                let selected_child_node = prior_node.children.iter_mut().find(|c| c.action == selected_action).unwrap();
                let selected_child_node_state = &mut selected_child_node.state;
                std::mem::replace(selected_child_node_state, MCTSNodeState::Expanded(index));
                drop(expanding_write_lock);

                // Flip the score in this case because we are going one node deeper and that viewpoint is from
                // the next player and not the current node's player.
                value_score = 1.0 - state_analysis.value_score;
                break;
            }

            // If the node is currently expanding. Wait for the expansion to be completed.
            if let MCTSNodeState::Expanding(expanding_lock) = selected_child_node_state {
                let expanding_lock = expanding_lock.clone();
                drop(node_write_lock);

                expanding_lock.read().await;

                // @TODO: go around again here to get expanded state
                continue;
            }

            if let MCTSNodeState::Expanded(expanded_node_index) = selected_child_node_state {
                // This case is when the node was not expanded initially when acquiring the original read lock.
                // Then when the write lock was acquired, the node was expanded in the interim. So proceed as if it
                // was always an expanded node.
                node_stack.push(*expanded_node_index);
                drop(node_write_lock);
            }
        }
    }

    // Reverse the value score at each depth according to the player's valuation perspective.
    for (i, node_index) in node_stack.into_iter().rev().enumerate() {
        let score = if i % 2 == 0 { value_score } else { 1.0 - value_score };
        let node_read_lock = &arena.read().await;
        let node = &node_read_lock[node_index];
        let mut W = node.W.lock().unwrap();
        *W = *W + score;
    }

    Ok(depth)
}



#[cfg(test)]
mod tests {
    use std::task::{Context,Poll};
    use std::pin::Pin;
    use std::future::Future;
    use super::*;
    use engine::game_state::{GameState};
    use model::analytics::{GameStateAnalysis};

    #[derive(Hash, PartialEq, Eq, Clone, Debug)]
    struct CountingGameState {
        pub p1_turn: bool,
        pub count: usize
    }

    impl CountingGameState {
        fn from_starting_count(p1_turn: bool, count: usize) -> Self {
            Self { p1_turn, count }
        }

        fn is_terminal_state(&self) -> Option<f32> {
            if self.count == 100 {
                Some(if self.p1_turn { 1.0 } else { -1.0 })
            } else if self.count == 0 {
                Some(if self.p1_turn { -1.0 } else { 1.0 })
            } else {
                None
            }
        }
    }

    impl GameState for CountingGameState {
        fn initial() -> Self {
            Self { p1_turn: true, count: 50 }
        }
    }

    struct CountingGameEngine {

    }

    impl CountingGameEngine {
        fn new() -> Self { Self {} }
    }

    #[derive(PartialEq, Eq, Clone, Debug)]
    enum CountingAction {
        Increment,
        Decrement,
        Stay
    }

    impl GameEngine for CountingGameEngine {
        type Action = CountingAction;
        type State = CountingGameState;

        fn take_action(&self, game_state: &Self::State, action: &Self::Action) -> Self::State {
            let count = game_state.count;

            let new_count = match action {
                CountingAction::Increment => count + 1,
                CountingAction::Decrement => count - 1,
                CountingAction::Stay => count
            };

            Self::State { p1_turn: !game_state.p1_turn, count: new_count }
        }

        fn is_terminal_state(&self, game_state: &Self::State) -> Option<f32> {
            game_state.is_terminal_state()
        }
    }

    struct CountingAnalytics {

    }

    impl CountingAnalytics {
        fn new() -> Self { Self {} }
    }

    struct CountingGameStateAnalysisFuture {
        output: Option<GameStateAnalysis<CountingAction>>
    }

    impl CountingGameStateAnalysisFuture {
        fn new(output: GameStateAnalysis<CountingAction>) -> Self {
            Self { output: Some(output) }
        }
    }

    impl Future for CountingGameStateAnalysisFuture {
        type Output = GameStateAnalysis<CountingAction>;

        fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
            Poll::Ready(self.get_mut().output.take().unwrap())
        }
    }

    impl GameAnalyzer for CountingAnalytics {
        type Action = CountingAction;
        type State = CountingGameState;
        type Future = CountingGameStateAnalysisFuture;

        fn get_state_analysis(&self, game_state: &Self::State) -> CountingGameStateAnalysisFuture {
            let count = game_state.count as f32;

            if let Some(score) = game_state.is_terminal_state() {
                return CountingGameStateAnalysisFuture::new(GameStateAnalysis {
                    policy_scores: Vec::new(),
                    value_score: score
                });
            }
            
            CountingGameStateAnalysisFuture::new(GameStateAnalysis {
                policy_scores: vec!(
                    ActionWithPolicy {
                        action: CountingAction::Increment,
                        policy_score: 0.3
                    },
                    ActionWithPolicy {
                        action: CountingAction::Decrement,
                        policy_score: 0.3
                    },
                    ActionWithPolicy {
                        action: CountingAction::Stay,
                        policy_score: 0.4
                    },
                ),
                value_score: (if game_state.p1_turn { count } else { 100.0 - count } / 50.0) - 1.0
            })
        }
    }

    #[tokio::test]
    async fn test_mcts_is_deterministic() {
        let game_state = CountingGameState::initial();
        let actions = List::new();
        let game_engine = CountingGameEngine::new();
        let analytics = CountingAnalytics::new();

        let mut mcts = MCTS::new(game_state.to_owned(), actions.to_owned(), &game_engine, &analytics, MCTSOptions::new(
            None,
            0.0,
            0.0,
            |_,_,_,_,_| 1.0,
            |_,_| 0.0
        ));

        let mut mcts2 = MCTS::new(game_state, actions, &game_engine, &analytics, MCTSOptions::new(
            None,
            0.0,
            0.0,
            |_,_,_,_,_| 1.0,
            |_,_| 0.0
        ));

        mcts.search(800).await.unwrap();
        mcts2.search(800).await.unwrap();

        let metrics = mcts.get_root_node_metrics().await.unwrap();
        let metrics2 = mcts.get_root_node_metrics().await.unwrap();

        assert_eq!(metrics, metrics2);
    }

    #[tokio::test]
    async fn test_mcts_chooses_best_p1_move() {
        let game_state = CountingGameState::initial();
        let actions = List::new();
        let game_engine = CountingGameEngine::new();
        let analytics = CountingAnalytics::new();

        let mut mcts = MCTS::new(game_state, actions, &game_engine, &analytics, MCTSOptions::new(
            None,
            0.0,
            0.0,
            |_,_,_,_,_| 1.0,
            |_,_| 0.0
        ));

        mcts.search(800).await.unwrap();
        let action = mcts.select_action().await.unwrap();

        assert_eq!(action, CountingAction::Increment);
    }

    #[tokio::test]
    async fn test_mcts_chooses_best_p2_move() {
        let game_state = CountingGameState::initial();
        let actions = List::new();
        let game_engine = CountingGameEngine::new();
        let analytics = CountingAnalytics::new();

        let mut mcts = MCTS::new(game_state, actions, &game_engine, &analytics, MCTSOptions::new(
            None,
            0.0,
            0.0,
            |_,_,_,_,_| 1.0,
            |_,_| 0.0
        ));

        mcts.advance_to_action(CountingAction::Increment).await.unwrap();

        mcts.search(800).await.unwrap();
        let action = mcts.select_action().await.unwrap();

        assert_eq!(action, CountingAction::Decrement);
    }

    #[tokio::test]
    async fn test_mcts_advance_to_next_works_without_search() {
        let game_state = CountingGameState::initial();
        let actions = List::new();
        let game_engine = CountingGameEngine::new();
        let analytics = CountingAnalytics::new();

        let mut mcts = MCTS::new(game_state, actions, &game_engine, &analytics, MCTSOptions::new(
            None,
            0.0,
            0.0,
            |_,_,_,_,_| 1.0,
            |_,_| 0.0
        ));

        mcts.advance_to_action(CountingAction::Increment).await.unwrap();
    }

    #[tokio::test]
    async fn test_mcts_metrics_returns_accurate_results() {
        let game_state = CountingGameState::initial();
        let actions = List::new();
        let game_engine = CountingGameEngine::new();
        let analytics = CountingAnalytics::new();

        let mut mcts = MCTS::new(game_state, actions, &game_engine, &analytics, MCTSOptions::new(
            None,
            0.0,
            0.0,
            |_,_,_,_,_| 1.0,
            |_,_| 0.0
        ));

        mcts.search(800).await.unwrap();

        let metrics = mcts.get_root_node_metrics().await.unwrap();

        assert_eq!(metrics, NodeMetrics {
            visits: 800,
            W: 400.88995,
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
        let actions = List::new();
        let game_engine = CountingGameEngine::new();
        let analytics = CountingAnalytics::new();

        let mut mcts = MCTS::new(game_state, actions, &game_engine, &analytics, MCTSOptions::new(
            None,
            0.0,
            0.0,
            |_,_,_,_,_| 1.0,
            |_,_| 0.0
        ));

        mcts.search(100).await.unwrap();

        let metrics = mcts.get_root_node_metrics().await.unwrap();

        assert_eq!(metrics, NodeMetrics {
            visits: 100,
            W: 50.040005,
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
        let actions = List::new();
        let game_engine = CountingGameEngine::new();
        let analytics = CountingAnalytics::new();

        let mut mcts = MCTS::new(game_state, actions, &game_engine, &analytics, MCTSOptions::new(
            None,
            0.0,
            0.0,
            |_,_,_,_,_| 1.0,
            |_,_| 0.0
        ));

        mcts.search(1).await.unwrap();

        let metrics = mcts.get_root_node_metrics().await.unwrap();

        assert_eq!(metrics, NodeMetrics {
            visits: 1,
            W: 0.5,
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
        let actions = List::new();
        let game_engine = CountingGameEngine::new();
        let analytics = CountingAnalytics::new();

        let mut mcts = MCTS::new(game_state, actions, &game_engine, &analytics, MCTSOptions::new(
            None,
            0.0,
            0.0,
            |_,_,_,_,_| 1.0,
            |_,_| 0.0
        ));

        mcts.search(2).await.unwrap();

        let metrics = mcts.get_root_node_metrics().await.unwrap();

        assert_eq!(metrics, NodeMetrics {
            visits: 2,
            W: 1.0,
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
        let actions = List::new();
        let game_engine = CountingGameEngine::new();
        let analytics = CountingAnalytics::new();

        let mut mcts = MCTS::new(game_state, actions, &game_engine, &analytics, MCTSOptions::new(
            None,
            0.0,
            0.0,
            |_,_,_,_,_| 0.1,
            |_,_| 0.0
        ));

        mcts.search(8000).await.unwrap();

        let metrics = mcts.get_root_node_metrics().await.unwrap();

        assert_eq!(metrics, NodeMetrics {
            visits: 8000,
            W: 6873.078,
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
        let actions = List::new();
        let game_engine = CountingGameEngine::new();
        let analytics = CountingAnalytics::new();

        let mut mcts = MCTS::new(game_state, actions, &game_engine, &analytics, MCTSOptions::new(
            None,
            0.0,
            0.0,
            |_,_,_,_,_| 1.0,
            |_,_| 0.0
        ));

        mcts.search(800).await.unwrap();

        let metrics = mcts.get_root_node_metrics().await.unwrap();

        assert_eq!(metrics, NodeMetrics {
            visits: 800,
            W: 8.879993,
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
        let actions = List::new();
        let game_engine = CountingGameEngine::new();
        let analytics = CountingAnalytics::new();

        let mut mcts = MCTS::new(game_state, actions, &game_engine, &analytics, MCTSOptions::new(
            None,
            0.0,
            0.0,
            |_,_,_,_,_| 1.0,
            |_,_| 0.0
        ));

        mcts.search(800).await.unwrap();

        let metrics = mcts.get_root_node_metrics().await.unwrap();

        assert_eq!(metrics, NodeMetrics {
            visits: 800,
            W: 784.65857,
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
        let actions = List::new();
        let game_engine = CountingGameEngine::new();
        let analytics = CountingAnalytics::new();
        let search_num_visits = 3;

        let mut non_clear_mcts = MCTS::new(game_state.clone(), actions.clone(), &game_engine, &analytics, MCTSOptions::new(
            None,
            0.0,
            0.0,
            |_,_,_,_,_| 1.0,
            |_,_| 0.0
        ));

        non_clear_mcts.search(search_num_visits).await.unwrap();
        let action = non_clear_mcts.select_action().await.unwrap();
        non_clear_mcts.advance_to_action_retain(action).await.unwrap();
        non_clear_mcts.search(search_num_visits).await.unwrap();

        let non_clear_metrics = non_clear_mcts.get_root_node_metrics().await.unwrap();

        let mut clear_mcts = MCTS::new(game_state, actions, &game_engine, &analytics, MCTSOptions::new(
            None,
            0.0,
            0.0,
            |_,_,_,_,_| 1.0,
            |_,_| 0.0
        ));

        clear_mcts.search(search_num_visits).await.unwrap();
        let action = clear_mcts.select_action().await.unwrap();
        clear_mcts.advance_to_action(action).await.unwrap();
        clear_mcts.search(search_num_visits).await.unwrap();

        let clear_metrics = clear_mcts.get_root_node_metrics().await.unwrap();

        assert_eq!(non_clear_metrics, clear_metrics);
    }
}
