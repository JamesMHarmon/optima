use rand::{ Rng };
use rand::seq::SliceRandom;

pub trait GameEngine<S, A> {
    fn get_state_analysis(&self, game_state: &S) -> GameStateAnalysis<A>;
    fn take_action(&self, game_state: &S, action: &A) -> S;
    fn temp(&self) -> S;
}

struct ActionWithPolicy<A> {
    action: A,
    policy_score: f64,
}

// If this is a terminal state, i.e. W/L/D. Return empty vector and value of -1, 0, or 1.
struct GameStateAnalysis<A> {
    policy_scores: Vec<ActionWithPolicy<A>>,
    value_score: f64
}

struct GameActionsAndState<S, A> {
    actions: GameActions<A>,
    state: S
}

type GameActions<A> = Vec<A>;

pub struct MCTSOptions<'a, S, R: Rng> {
    dirichlet_alpha: f64,
    dirichlet_epsilon: f64,
    cpuct: &'a Fn(&S) -> f64,
    temperature: &'a Fn(&S) -> f64,
    rng: R
}

pub struct MCTS<'a, S, A, E: GameEngine<S, A>, R: Rng> {
    options: MCTSOptions<'a, S, R>,
    game_engine: &'a E,
    starting_game_state: S,
    root: Option<MCTSNode<S, A>>,
}

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
    node: &'a MCTSChildNode<S, A>,
    score: f64
}

struct StateAnalysisValue {
    value: f64
}

impl<'a, S, A, E: GameEngine<S, A>, R: Rng> MCTS<'a, S, A, E, R> {
    pub fn new(game_state: S, game_engine: &'a E, options: MCTSOptions<'a, S, R>) -> Self {
        MCTS {
            options,
            game_engine,
            starting_game_state: game_state,
            root: None
        }
    }

    pub fn get_next_action(&mut self, number_of_nodes_to_search: usize) -> Result<A, &'static str> {
        let root_node = self.get_or_create_root_node();

        for _ in 0..number_of_nodes_to_search {
            MCTS::<S, A, E, R>::recurse_path_and_expand(root_node);
        }

        let current_root = self.root.take().ok_or("No root node found!")?;
        let most_visited_node = self.take_most_visited_node(current_root)?;

        // Update the tree now that this is the next root node.
        self.root = most_visited_node.node;
        Ok(most_visited_node.action)
    }

    fn take_most_visited_node(&mut self, current_root: MCTSNode<S, A>) -> Result<MCTSChildNode<S, A>, &'static str> {
        let candidate_nodes = current_root.children;
        let visited_nodes: Vec<MCTSChildNode<S, A>> = candidate_nodes.into_iter().filter(|n| { n.node.is_some() }).collect();
        let max_visits = visited_nodes.iter().map(|n| { n.node.as_ref().unwrap().visits }).max().expect("No visited_nodes to choose from");
        let mut max_nodes: Vec<MCTSChildNode<S, A>> = visited_nodes.into_iter().filter(|n| {
            n.node.as_ref().map_or(false, |n| { n.visits >= max_visits })
        }).collect();

        match max_nodes.len() {
            0 => Err("No candidate moves available"),
            1 => Ok(max_nodes.remove(0)),
            _ => {
                let random_idx = self.options.rng.gen_range(0, max_nodes.len());
                Ok(max_nodes.remove(random_idx))
            }
        }
    }

    fn recurse_path_and_expand(node: &mut MCTSNode<S, A>) -> StateAnalysisValue {
        let selected_child_node = MCTS::<S, A, E, R>::select_path_using_PUCT(node);

        let result = match &selected_child_node.node {
            None => {
                let (expanded_node, state_analysis) = MCTS::<S, A, E, R>::expand_leaf(&node.game_state, &selected_child_node.action);
                selected_child_node.node = Some(expanded_node);
                state_analysis
            },
            // @TODO: Simplify this so that Some(_) is a reference as mut.
            Some(_) => MCTS::<S, A, E, R>::recurse_path_and_expand(&mut selected_child_node.node.as_mut().unwrap())
        };

        node.visits += 1;
        node.W += result.value;
        result
    }

    fn select_path_using_PUCT(node: &mut MCTSNode<S, A>) -> &mut MCTSChildNode<S, A> {
        panic!()
        // // @TODO: Add temperature
        // for puct in  MCTS::<S, A, E, R>::get_PUCT_for_nodes(node) {

        // }

        // // @TODO: Update this to get either max or by dirichlet noise?
        // & MCTS::<S, A, E, R>::get_PUCT_for_nodes(node)[0].node
    }

    //@TODO: See if we can return an iterator here.
    fn get_PUCT_for_nodes(&self, node: &'a MCTSNode<S, A>) -> Vec<NodePUCT<'a, S, A>> {
        panic!()
        // let node_children = node.children;
        // let game_state = node.game_state;

        // // @TODO: See if this is equivalent to (node.visits - 1).
        // let Nsb: usize = node_children.iter()
        //     .filter_map(|n| { n.node })
        //     .map(|n| { n.visits })
        //     .sum();

        // node_children.iter().map(|child| {
        //     let child_node = child.node;
        //     let Psa = child.policy_score;
        //     let Nsa = child_node.map_or(0, |n| { n.visits });
        //     // Should the child's game state be passed here instead?
        //     let cpuct = (self.options.cpuct)(&game_state);
        //     let Usa = cpuct * Psa * (Nsb as f64).sqrt() / (1 + Nsa) as f64;

        //     let Qsa = child_node.map_or(0.0, |n| { n.W / n.visits as f64 });

        //     let PUCT = Qsa + Usa;

        //     NodePUCT {
        //         node: &child,
        //         score: PUCT
        //     }
        // }).collect()
    }

    fn expand_leaf(game_state: &S, action: &A) -> (MCTSNode<S, A>, StateAnalysisValue) {
        panic!()
        // let updated_game_state = self.game_engine.take_action(game_state, action);
        // let analysis_result = self.game_engine.get_state_analysis(&updated_game_state);

        // (
        //     MCTSNode::new(updated_game_state, analysis_result.value_score, &analysis_result.policy_scores),
        //     StateAnalysisValue { value: analysis_result.value_score }
        // )
    }

    fn get_or_create_root_node(&mut self) -> &mut MCTSNode<S, A> {
        let game_engine = self.game_engine;
        self.root.get_or_insert_with(|| MCTSNode {
            visits: 0,
            W: 0.0,
            game_state: game_engine.temp(),
            children: Vec::new()
        })
    }

    fn create_root_node(&self, game_state: S) -> MCTSNode<S, A> {
        panic!()
        // let analysis_result = self.game_engine.get_state_analysis(&game_state);

        // MCTSNode::new(game_state, analysis_result.value_score, &analysis_result.policy_scores)
    }
}

impl<S, A> MCTSNode<S, A> {
    pub fn new(game_state: S, value_score: f64, policy_scores: &Vec<ActionWithPolicy<A>>) -> Self {
        panic!()
        // MCTSNode {
        //     visits: 1,
        //     W: value_score,
        //     game_state: game_state,
        //     children: policy_scores.iter().map(|action_with_policy| {
        //         MCTSChildNode {
        //             action: action_with_policy.action,
        //             policy_score: action_with_policy.policy_score,
        //             node: None
        //         }
        //     }).collect()
        // }
    }
}