use rand::{ Rng };
use super::engine::{GameEngine};
use super::analysis::{ActionWithPolicy};

pub struct MCTSOptions<'a, S, R: Rng> {
    dirichlet_alpha: f64,
    dirichlet_epsilon: f64,
    cpuct: &'a dyn Fn(&S) -> f64,
    temperature: &'a dyn Fn(&S) -> f64,
    rng: R
}

impl<'a, S, R: Rng> MCTSOptions<'a, S, R> {
    pub fn new(
        dirichlet_alpha: f64,
        dirichlet_epsilon: f64,
        cpuct: &'a dyn Fn(&S) -> f64,
        temperature: &'a dyn Fn(&S) -> f64,
        rng: R
    ) -> Self {
        MCTSOptions {
            dirichlet_alpha,
            dirichlet_epsilon,
            cpuct,
            temperature,
            rng,
        }
    }
}

pub struct MCTS<'a, S, A, E: GameEngine<S, A>, R: Rng> {
    options: MCTSOptions<'a, S, R>,
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

impl<'a, S, A, E: GameEngine<S, A>, R: Rng> MCTS<'a, S, A, E, R> where E: 'a {
    pub fn new(game_state: S, game_engine: &'a E, options: MCTSOptions<'a, S, R>) -> Self {
        MCTS {
            options,
            game_engine,
            starting_game_state: Some(game_state),
            root: None
        }
    }

    pub fn get_next_action(&mut self, number_of_nodes_to_search: usize) -> Result<(A, usize), &'static str> {
        let game_engine = self.game_engine;
        let cpuct = self.options.cpuct;
        let root_node = self.get_or_create_root_node();
        let mut max_depth: usize = 0;

        for _ in 0..number_of_nodes_to_search {
            let (_, md) = MCTS::<S, A, E, R>::recurse_path_and_expand(root_node, game_engine, cpuct, 0);
            max_depth = md;
        }

        let current_root = self.root.take().ok_or("No root node found!")?;
        let most_visited_node = self.take_most_visited_node(current_root)?;

        // Update the tree now that this is the next root node.
        self.root = most_visited_node.node;
        Ok((most_visited_node.action, max_depth))
    }

    fn take_most_visited_node(&mut self, current_root: MCTSNode<S, A>) -> Result<MCTSChildNode<S, A>, &'static str> {
        let candidate_nodes = current_root.children;
        let visited_nodes: Vec<MCTSChildNode<S, A>> = candidate_nodes.into_iter().filter(|n| { n.node.is_some() }).collect();
        let max_visits = visited_nodes.iter().map(|n| { n.node.as_ref().unwrap().visits }).max().expect("No visited_nodes to choose from");
        
        for node in visited_nodes.iter() {
            println!("Visits: {:?}, W: {:?}", node.node.as_ref().unwrap().visits, node.policy_score);
        }
        
        let mut max_nodes: Vec<MCTSChildNode<S, A>> = visited_nodes.into_iter().filter(|n| {
            n.node.as_ref().map_or(false, |n| { n.visits >= max_visits })
        }).collect();

        match max_nodes.len() {
            0 => Err("No candidate moves available"),
            1 => Ok(max_nodes.remove(0)),
            _ => {
                println!("Random!");
                let random_idx = self.options.rng.gen_range(0, max_nodes.len());
                Ok(max_nodes.remove(random_idx))
            }
        }
    }

    fn recurse_path_and_expand(node: &mut MCTSNode<S, A>, game_engine: &E, cpuct: &dyn Fn(&S) -> f64, depth: usize) -> (StateAnalysisValue, usize) {
        let game_state = &node.game_state;
        let selected_child_node = MCTS::<S, A, E, R>::select_path_using_PUCT(&mut node.children, game_state, cpuct);

        let (result, depth) = match &mut selected_child_node.node {
            None => {
                let (expanded_node, state_analysis) = MCTS::<S, A, E, R>::expand_leaf(
                    game_state,
                    &selected_child_node.action,
                    game_engine
                );
                selected_child_node.node = Some(expanded_node);
                (state_analysis, depth)
            },
            Some(node) => MCTS::<S, A, E, R>::recurse_path_and_expand(node, game_engine, cpuct, depth + 1)
        };

        node.visits += 1;
        node.W += result.value;
        (result, depth)
    }

    #[allow(non_snake_case)]
    fn select_path_using_PUCT(nodes: &'a mut Vec<MCTSChildNode<S, A>>, game_state: &S, cpuct: &dyn Fn(&S) -> f64) -> &'a mut MCTSChildNode<S, A> {
        // @TODO: Add temperature
        // for puct in  MCTS::<S, A, E, R>::get_PUCT_for_nodes(node, cpuct) {
        //     return puct.node
        // }

        // @TODO: Update this to get either max or by dirichlet noise?
        // @TODO: Update to randomize if multiple values are max.
        // @TODO: Combine method with take most visited node?
        let mut pucts = MCTS::<S, A, E, R>::get_PUCT_for_nodes(nodes, game_state, cpuct);
        let initial = pucts.remove(pucts.len() - 1);
        let mut max_puct_score = initial.score;
        let mut best_node = initial.node;

        for puct in pucts {
            if puct.score > max_puct_score {
                max_puct_score = puct.score;
                best_node = puct.node;
            }
        }

        best_node
    }

    //@TODO: See if we can return an iterator here.
    #[allow(non_snake_case)]
    fn get_PUCT_for_nodes(nodes: &'a mut Vec<MCTSChildNode<S, A>>, game_state: &S, cpuct: &dyn Fn(&S) -> f64) -> Vec<NodePUCT<'a, S, A>> {
        // @TODO: See if this is equivalent to (node.visits - 1).
        let Nsb: usize = nodes.iter()
            .filter_map(|n| { n.node.as_ref() })
            .map(|n| { n.visits })
            .sum();

        nodes.iter_mut().map(|child| {
            let child_node = &child.node;
            let Psa = child.policy_score;
            let Nsa = child_node.as_ref().map_or(0, |n| { n.visits });
            // Should the child's game state be passed here instead?
            let cpuct = cpuct(&game_state);
            let Usa = cpuct * Psa * (Nsb as f64).sqrt() / (1 + Nsa) as f64;

            let Qsa = child_node.as_ref().map_or(0.0, |n| { n.W / n.visits as f64 });

            let PUCT = Qsa + Usa;

            NodePUCT {
                node: child,
                score: PUCT
            }
        }).collect()
    }

    fn expand_leaf(game_state: &S, action: &A, game_engine: &E) -> (MCTSNode<S, A>, StateAnalysisValue) {
        let new_game_state = game_engine.take_action(game_state, action);
        let analysis_result = game_engine.get_state_analysis(&new_game_state);

        (
            MCTSNode::new(new_game_state, analysis_result.value_score, analysis_result.policy_scores),
            StateAnalysisValue { value: analysis_result.value_score }
        )
    }

    fn get_or_create_root_node(&mut self) -> &mut MCTSNode<S, A> {
        let starting_game_state = self.starting_game_state.take();
        let game_engine = self.game_engine;

        self.root.get_or_insert_with(|| {
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