use rand::{ Rng };
use rand::prelude::Distribution;
use rand::distributions::{Dirichlet,WeightedIndex};

use super::engine::{GameEngine};
use super::analysis::{ActionWithPolicy};

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

pub struct MCTS<'a, S, A, E: GameEngine<S, A>, R: Rng> {
    options: MCTSOptions<'a, S, A, R>,
    game_engine: &'a E,
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
    value: f64
}

#[allow(non_snake_case)]
impl<'a, S, A, E: GameEngine<S, A>, R: Rng> MCTS<'a, S, A, E, R> where E: 'a {
    pub fn new(game_state: S, game_engine: &'a E, options: MCTSOptions<'a, S, A, R>) -> Self {
        MCTS {
            options,
            game_engine,
            starting_game_state: Some(game_state),
            root: None
        }
    }

    pub fn get_next_action(&mut self, number_of_playouts: usize) -> Result<(A, usize), &'static str> {
        let game_engine = self.game_engine;
        let dirichlet = &self.options.dirichlet;
        let cpuct = self.options.cpuct;
        let temp = self.options.temperature;
        let rng = &mut self.options.rng;
        let root = &mut self.root;
        let starting_game_state = &mut self.starting_game_state;
        let root_node = MCTS::<S, A, E, R>::get_or_create_root_node(root, starting_game_state, game_engine);
        let mut max_depth: usize = 0;

        for _ in 0..number_of_playouts {
            let (_, md) = MCTS::<S, A, E, R>::recurse_path_and_expand(root_node, game_engine, cpuct, temp, dirichlet, rng, 0)?;
            max_depth = md;
        }

        let current_root = self.root.take().ok_or("No root node found!")?;
        let most_visited_node = MCTS::<S, A, E, R>::take_most_visited_node(current_root, rng)?;

        // Update the tree now that this is the next root node.
        self.root = Some(most_visited_node.1);
        Ok((most_visited_node.0, max_depth))
    }

    fn take_most_visited_node(current_root: MCTSNode<S, A>, rng: &mut R) -> Result<(A, MCTSNode<S, A>), &'static str> {
        let visited_nodes: Vec<(A, MCTSNode<S, A>)> = current_root.children.into_iter()
            .filter_map(|n| {
                let action = n.action;
                n.node.map(|node| (action, node))
            })
            .collect();

        for visited_node in visited_nodes.iter() {
            println!("{}, {}", visited_node.1.visits, visited_node.1.W);
        }

        let max_visits = visited_nodes.iter().map(|n| { n.1.visits }).max().ok_or("No visited_nodes to choose from")?;

        let mut max_nodes: Vec<(A, MCTSNode<S, A>)> = visited_nodes.into_iter().filter(|n| n.1.visits >= max_visits).collect();

        let chosen_idx = match max_nodes.len() {
            0 => Err("No candidate moves available"),
            1 => Ok(0),
            len => Ok(rng.gen_range(0, len))
        };

        chosen_idx.map(|idx| max_nodes.remove(idx))
    }

    fn recurse_path_and_expand(node: &mut MCTSNode<S, A>, game_engine: &E, cpuct: Cpuct<S, A>, temp: Temp<S>, dirichlet: &Option<DirichletOptions>, rng: &mut R, depth: usize) -> Result<(StateAnalysisValue, usize), &'static str> {
        let game_state = &node.game_state;
        let Nsb = node.visits - 1;

        let selected_child_node = MCTS::<S, A, E, R>::select_path_using_PUCT(&mut node.children, Nsb, game_state, cpuct, temp, rng)?;

        let (result, depth) = match &mut selected_child_node.node {
            None => {
                let (expanded_node, state_analysis) = MCTS::<S, A, E, R>::expand_leaf(
                    game_state,
                    &selected_child_node.action,
                    game_engine,
                    dirichlet,
                    rng
                );
                selected_child_node.node = Some(expanded_node);
                (state_analysis, depth)
            },
            Some(node) => MCTS::<S, A, E, R>::recurse_path_and_expand(node, game_engine, cpuct, temp, &None, rng, depth + 1)?
        };

        node.visits += 1;
        node.W += result.value;
        Ok((result, depth))
    }

    fn select_path_using_PUCT(nodes: &'a mut Vec<MCTSChildNode<S, A>>, Nsb: usize, game_state: &S, cpuct: Cpuct<S, A>, temp: Temp<S>, rng: &mut R) -> Result<&'a mut MCTSChildNode<S, A>, &'static str> {
        let mut pucts = MCTS::<S, A, E, R>::get_PUCT_for_nodes(nodes, Nsb, game_state, cpuct);

        let temp = temp(game_state);
        let chosen_puct_idx = if temp == 0.0 {
            MCTS::<S, A, E, R>::select_path_using_PUCT_max(&pucts, rng)
        } else {
            MCTS::<S, A, E, R>::select_path_using_PUCT_Temperature(&pucts, temp, rng)
        }?;

        Ok(pucts.remove(chosen_puct_idx).node)
    }

    fn select_path_using_PUCT_max(pucts: &Vec<NodePUCT<S, A>>, rng: &mut R) -> Result<usize, &'static str> {
        let max_puct = pucts.iter().fold(std::f64::MIN, |acc, puct| f64::max(acc, puct.score));
        let mut max_nodes: Vec<usize> = pucts.into_iter()
            .enumerate()
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

    //@TODO: See if we can return an iterator here.
    fn get_PUCT_for_nodes(nodes: &'a mut Vec<MCTSChildNode<S, A>>, Nsb: usize, game_state: &S, cpuct: Cpuct<S, A>) -> Vec<NodePUCT<'a, S, A>> {
        nodes.iter_mut().map(|child| {
            let child_node = &child.node;
            let Psa = child.policy_score;
            let Nsa = child_node.as_ref().map_or(0, |n| { n.visits });

            let cpuct = cpuct(&game_state, &child.action);
            let root_Nsb = if Nsb == 0 { 1.0 } else {  ((Nsb + 1) as f64).sqrt() };
            let Usa = cpuct * Psa * root_Nsb / (1 + Nsa) as f64;

            let Qsa = child_node.as_ref().map_or(0.0, |n| { n.W / n.visits as f64 });

            let PUCT = Qsa + Usa;

            NodePUCT {
                node: child,
                score: PUCT
            }
        }).collect()
    }

    fn expand_leaf(game_state: &S, action: &A, game_engine: &E, dirichlet: &Option<DirichletOptions>, rng: &mut R) -> (MCTSNode<S, A>, StateAnalysisValue) {
        let new_game_state = game_engine.take_action(game_state, action);
        let analysis_result = game_engine.get_state_analysis(&new_game_state);

        let pucts: Vec<ActionWithPolicy<A>> = match dirichlet {
            Some(dirichlet) => {
                let e = dirichlet.epsilon;
                let pucts = analysis_result.policy_scores;
                let dirichlet_noise = Dirichlet::new_with_param(dirichlet.alpha, pucts.len()).sample(rng);
                dirichlet_noise.into_iter().zip(pucts).map(|(noise, awp)| ActionWithPolicy {
                    action: awp.action,
                    policy_score: (1.0 - e) * awp.policy_score + e * noise
                }).collect()
            },
            None => analysis_result.policy_scores
        };

        (
            MCTSNode::new(new_game_state, analysis_result.value_score, pucts),
            StateAnalysisValue { value: analysis_result.value_score }
        )
    }

    fn get_or_create_root_node(root: &'a mut Option<MCTSNode<S, A>>, starting_game_state: &mut Option<S>, game_engine: &E) -> &'a mut MCTSNode<S, A> {
        let starting_game_state = starting_game_state.take();
        let game_engine = game_engine;

        root.get_or_insert_with(|| {
            MCTS::<S, A, E, R>::create_root_node(
                starting_game_state.expect("Tried to use the same starting game state twice"),
                game_engine
            )
        })
    }

    fn create_root_node(game_state: S, game_engine: &E) -> MCTSNode<S, A> {
        let analysis_result = game_engine.get_state_analysis(&game_state);

        MCTSNode::new(game_state, analysis_result.value_score, analysis_result.policy_scores)
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