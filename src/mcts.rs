use rand::{ Rng };
use rand::prelude::Distribution;
use rand::distributions::{Dirichlet,WeightedIndex};

use super::game_state::GameState;
use super::engine::{GameEngine};
use super::analytics::{ActionWithPolicy,GameAnalytics};
use super::node_metrics::{NodeMetrics};

type Cpuct<'a, S, A> = &'a dyn Fn(&S, &A) -> f64;
type Temp<'a, S> = &'a dyn Fn(&S) -> f64;

pub struct DirichletOptions {
    pub alpha: f64,
    pub epsilon: f64
}

pub struct MCTSOptions<'a, S, A, R: Rng> {
    dirichlet: Option<DirichletOptions>,
    cpuct: Cpuct<'a, S, A>,
    temperature: Temp<'a, S>,
    rng: R
}

impl<'a, S, A, R: Rng> MCTSOptions<'a, S, A, R> {
    pub fn new(
        dirichlet: Option<DirichletOptions>,
        cpuct: Cpuct<'a, S, A>,
        temperature: Temp<'a, S>,
        rng: R
    ) -> Self {
        MCTSOptions {
            dirichlet,
            cpuct,
            temperature,
            rng,
        }
    }
}

pub struct MCTS<'a, S, A, E, M, R>
where
    S: GameState,
    A: Clone + Eq,
    E: GameEngine,
    M: GameAnalytics,
    R: Rng
{
    options: MCTSOptions<'a, S, A, R>,
    game_engine: &'a E,
    analytics: &'a M,
    starting_game_state: Option<S>,
    root: Option<MCTSNode<S, A>>,
}


#[allow(non_snake_case)]
struct MCTSNode<S, A> {
    visits: usize,
    W: f64,
    game_state: S,
    children: Vec<MCTSChildNode<S, A>>
}

struct MCTSChildNode<S, A> {
    action: A,
    policy_score: f64,
    node: Option<MCTSNode<S, A>>
}

struct NodePUCT<'a, S, A> {
    node: &'a mut MCTSChildNode<S, A>,
    score: f64
}

struct StateAnalysisValue {
    value_score: f64
}

#[allow(non_snake_case)]
impl<'a, S, A, E, M, R> MCTS<'a, S, A, E, M, R>
where
    S: GameState,
    A: Clone + Eq,
    E: 'a + GameEngine<State=S,Action=A>,
    M: 'a + GameAnalytics<State=S,Action=A>,
    R: Rng
{
    pub fn new(
        game_state: S,
        game_engine: &'a E,
        analytics: &'a M,
        options: MCTSOptions<'a, S, A, R>
    ) -> Self {
        MCTS {
            options,
            game_engine,
            analytics,
            starting_game_state: Some(game_state),
            root: None
        }
    }

    pub fn search(&mut self, visits: usize) -> Result<(A, usize), &'static str> {
        let game_engine = &self.game_engine;
        let cpuct = self.options.cpuct;
        let temp = self.options.temperature;
        let dirichlet = &self.options.dirichlet;
        let analytics = &mut self.analytics;
        let rng = &mut self.options.rng;
        let root = &mut self.root;
        let starting_game_state = &mut self.starting_game_state;
        let mut root_node = MCTS::<S, A, E, M, R>::get_or_create_root_node(root, starting_game_state, analytics);
        let mut max_depth: usize = 0;

        Self::apply_dirichlet_noise_to_node(&mut root_node, dirichlet, rng);

        while root_node.visits < visits {
            let (_, md) = Self::recurse_path_and_expand(root_node, game_engine, analytics, cpuct, temp, rng, 0)?;

            if md > max_depth {
                max_depth = md;
            }
        }

        let most_visited_action = Self::get_most_visited_action(&root_node, rng)?;

        Ok((most_visited_action, max_depth))
    }

    pub fn advance_to_action(&mut self, action: &A) -> Result<(), &'static str> {
        let mut current_root = self.root.take().ok_or("No root node found!")?;
        let node = Self::take_node_of_action(&mut current_root, action)?;
        self.root = Some(node.unwrap_or_else(|| {
            Self::expand_leaf(&current_root.game_state, action, self.game_engine, &mut self.analytics).0
        }));

        Ok(())
    }

    pub fn get_root_node_metrics(&self) -> Result<NodeMetrics<A>, &'static str> {
        let root = self.root.as_ref().ok_or("No root node found!")?;

        Ok(NodeMetrics {
            visits: root.visits,
            W: root.W,
            children_visits: root.children.iter().map(|n| (
                n.action.clone(),
                n.node.as_ref().map_or(0, |n| n.visits)
            )).collect()
        })
    }

    fn get_most_visited_action(current_root: &MCTSNode<S, A>, rng: &mut R) -> Result<A, &'static str> {
        let max_visits = current_root.children.iter()
            .map(|n| n.node.as_ref().map_or(0, |n| n.visits))
            .max().ok_or("No visited_nodes to choose from")?;

        let mut max_actions: Vec<A> = current_root.children.iter()
            .filter_map(|n| {
                if n.node.as_ref().map_or(0, |n| n.visits) >= max_visits {
                    Some(n.action.clone())
                } else {
                    None
                }
            })
            .collect();

        let chosen_idx = match max_actions.len() {
            0 => Err("No candidate moves available"),
            1 => Ok(0),
            len => Ok(rng.gen_range(0, len))
        }?;

        Ok(max_actions.remove(chosen_idx))
    }

    fn take_node_of_action(current_root: &mut MCTSNode<S, A>, action: &A) -> Result<Option<MCTSNode<S, A>>, &'static str> {
        let matching_action = current_root.children.iter_mut().find(|n| n.action == *action).ok_or("No matching Action")?;

        Ok(matching_action.node.take())
    }

    fn recurse_path_and_expand(node: &mut MCTSNode<S, A>, game_engine: &E, analytics: &M, cpuct: Cpuct<S, A>, temp: Temp<S>, rng: &mut R, depth: usize) -> Result<(StateAnalysisValue, usize), &'static str> {
        // If the node is a terminal node.
        if node.children.len() == 0 {
            node.visits += 1;
            return Ok((StateAnalysisValue { value_score: node.W }, depth));
        }

        let game_state = &node.game_state;
        let Nsb = node.visits;
        let selected_child_node = MCTS::<S, A, E, M, R>::select_path_using_PUCT(&mut node.children, Nsb, game_state, cpuct, temp, rng)?;

        let (result, depth) = match &mut selected_child_node.node {
            None => {
                let (expanded_node, state_analysis) = MCTS::<S, A, E, M, R>::expand_leaf(
                    game_state,
                    &selected_child_node.action,
                    game_engine,
                    analytics
                );
                selected_child_node.node = Some(expanded_node);
                (state_analysis, depth)
            },
            Some(node) => MCTS::<S, A, E, M, R>::recurse_path_and_expand(node, game_engine, analytics, cpuct, temp,rng, depth + 1)?
        };

        // Reverse the value score since the value is of the child nodes evaluation, which is the other player.
        let result = StateAnalysisValue { value_score: 1.0 - result.value_score };

        node.visits += 1;
        node.W += result.value_score;
        Ok((result, depth))
    }

    fn select_path_using_PUCT(nodes: &'a mut Vec<MCTSChildNode<S, A>>, Nsb: usize, game_state: &S, cpuct: Cpuct<S, A>, temp: Temp<S>, rng: &mut R) -> Result<&'a mut MCTSChildNode<S, A>, &'static str> {
        let mut pucts = Self::get_PUCT_for_nodes(nodes, Nsb, game_state, cpuct);

        let temp = temp(game_state);
        let chosen_puct_idx = if temp == 0.0 {
            Self::select_path_using_PUCT_max(&pucts, rng)
        } else {
            Self::select_path_using_PUCT_Temperature(&pucts, temp, rng)
        }?;

        Ok(pucts.remove(chosen_puct_idx).node)
    }

    fn select_path_using_PUCT_max(pucts: &Vec<NodePUCT<S, A>>, rng: &mut R) -> Result<usize, &'static str> {
        let max_puct = pucts.iter().fold(std::f64::MIN, |acc, puct| f64::max(acc, puct.score));
        let mut max_nodes: Vec<usize> = pucts.into_iter().enumerate()
            .filter_map(|(i, puct)| if puct.score >= max_puct { Some(i) } else { None })
            .collect();

        match max_nodes.len() {
            0 => Err("No candidate moves available"),
            1 => Ok(max_nodes.remove(0)),
            len => Ok(max_nodes.remove(rng.gen_range(0, len)))
        }
    }

    fn select_path_using_PUCT_Temperature(pucts: &Vec<NodePUCT<S, A>>, temp: f64, rng: &mut R) -> Result<usize, &'static str> {
        let puct_scores = pucts.iter().map(|puct| puct.score.powf(1.0 / temp));

        let weighted_index = WeightedIndex::new(puct_scores).map_err(|_| "Invalid puct scores")?;

        let chosen_idx = weighted_index.sample(rng);
        Ok(chosen_idx)
    }

    fn get_PUCT_for_nodes(nodes: &'a mut Vec<MCTSChildNode<S, A>>, Nsb: usize, game_state: &S, cpuct: Cpuct<S, A>) -> Vec<NodePUCT<'a, S, A>> {
        nodes.iter_mut().map(|child| {
            let child_node = &child.node;
            let Psa = child.policy_score;
            let Nsa = child_node.as_ref().map_or(0, |n| { n.visits });

            let cpuct = cpuct(&game_state, &child.action);
            let root_Nsb = (Nsb as f64).sqrt();
            let Usa = cpuct * Psa * root_Nsb / (1 + Nsa) as f64;

            let Qsa = child_node.as_ref().map_or(0.0, |n| { n.W / n.visits as f64 });

            let PUCT = Qsa + Usa;

            NodePUCT {
                node: child,
                score: PUCT
            }
        }).collect()
    }

    fn expand_leaf(game_state: &S, action: &A, game_engine: &E, analytics: &M) -> (MCTSNode<S, A>, StateAnalysisValue) {
        let new_game_state = game_engine.take_action(game_state, action);
        MCTS::<S, A, E, M, R>::analyse_and_create_node(new_game_state, analytics)
    }

    fn get_or_create_root_node(
        root: &'a mut Option<MCTSNode<S, A>>,
        starting_game_state: &mut Option<S>,
        analytics: &M
    ) -> &'a mut MCTSNode<S, A> {
        let starting_game_state = starting_game_state.take();

        root.get_or_insert_with(|| {
            Self::analyse_and_create_node(
                starting_game_state.expect("Tried to use the same starting game state twice"),
                analytics
            ).0
        })
    }

    // Value range is [-1, 1] for the "get_state_analysis" method. However internally for the MCTS a range of
    // [0, 1] is used.
    fn analyse_and_create_node(game_state: S, analytics: &M) -> (MCTSNode<S, A>, StateAnalysisValue) {
        let analysis_result = analytics.get_state_analysis(&game_state);

        let value_score = (analysis_result.value_score + 1.0) / 2.0;

        (
            MCTSNode::new(game_state, value_score, analysis_result.policy_scores),
            StateAnalysisValue { value_score }
        )
    }

    fn apply_dirichlet_noise_to_node(node: &mut MCTSNode<S, A>, dirichlet: &Option<DirichletOptions>, rng: &mut R) {
        if let Some(dirichlet) = dirichlet {
            let policy_scores: Vec<f64> = node.children.iter().map(|child_node| {
                child_node.policy_score
            }).collect();

            let updated_policy_scores = MCTS::<S, A, E, M, R>::apply_dirichlet_noise(policy_scores, dirichlet, rng);

            for (child, policy_score) in node.children.iter_mut().zip(updated_policy_scores.into_iter()) {
                child.policy_score = policy_score;
            }
        }
    }

    fn apply_dirichlet_noise(policy_scores: Vec<f64>, dirichlet: &DirichletOptions, rng: &mut R) -> Vec<f64>
    {
        // Do not apply noise if there is only one action.
        if policy_scores.len() < 2 {
            return policy_scores;
        }

        let e = dirichlet.epsilon;
        let dirichlet_noise = Dirichlet::new_with_param(dirichlet.alpha, policy_scores.len()).sample(rng);

        dirichlet_noise.into_iter().zip(policy_scores).map(|(noise, policy_score)|
            (1.0 - e) * policy_score + e * noise
        ).collect()
    }
}

impl<S, A> MCTSNode<S, A> {
    pub fn new(game_state: S, value_score: f64, policy_scores: Vec<ActionWithPolicy<A>>) -> Self {
        MCTSNode {
            visits: 1,
            W: value_score,
            game_state: game_state,
            children: policy_scores.into_iter().map(|action_with_policy| {
                MCTSChildNode {
                    action: action_with_policy.action,
                    policy_score: action_with_policy.policy_score,
                    node: None
                }
            }).collect()
        }
    }
}