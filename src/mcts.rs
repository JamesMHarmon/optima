use std::fmt::Debug;
use rand::{ Rng };
use rand::prelude::Distribution;
use rand::distributions::{Dirichlet,WeightedIndex};

use super::game_state::GameState;
use super::engine::{GameEngine};
use super::analytics::{ActionWithPolicy,GameAnalytics};
use super::node_metrics::{NodeMetrics};

// pub type Cpuct<S, A> = dyn Fn(&S, &A) -> f64;
// pub type Temp<S> = dyn Fn(&S) -> f64;

pub struct DirichletOptions {
    pub alpha: f64,
    pub epsilon: f64
}

pub struct MCTSOptions<R>
where
    R: Rng
{
    dirichlet: Option<DirichletOptions>,
    cpuct: f64,
    temperature: f64,
    rng: R
}

impl<R> MCTSOptions<R>
where
    R: Rng
{
    pub fn new(
        dirichlet: Option<DirichletOptions>,
        cpuct: f64,
        temperature: f64,
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
    A: Clone + Eq + Debug,
    E: GameEngine,
    M: GameAnalytics,
    R: Rng
{
    options: MCTSOptions<R>,
    game_engine: &'a E,
    analytics: &'a M,
    starting_game_state: Option<S>,
    root: Option<MCTSNode<S, A>>,
}


#[allow(non_snake_case)]
#[derive(Debug)]
struct MCTSNode<S, A> {
    visits: usize,
    W: f64,
    game_state: S,
    children: Vec<MCTSChildNode<S, A>>
}

#[derive(Debug)]
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
    A: Clone + Eq + Debug,
    E: 'a + GameEngine<State=S,Action=A>,
    M: 'a + GameAnalytics<State=S,Action=A>,
    R: Rng
{
    pub fn new(
        game_state: S,
        game_engine: &'a E,
        analytics: &'a M,
        options: MCTSOptions<R>
    ) -> Self {
        MCTS {
            options,
            game_engine,
            analytics,
            starting_game_state: Some(game_state),
            root: None
        }
    }

    pub async fn search(&mut self, visits: usize) -> Result<(A, usize), &'static str> {
        let game_engine = &self.game_engine;
        let cpuct = self.options.cpuct;
        let temp = self.options.temperature;
        let dirichlet = &self.options.dirichlet;
        let rng = &mut self.options.rng;
        let analytics = &mut self.analytics;
        let root = &mut self.root;
        let starting_game_state = &mut self.starting_game_state;
        let mut root_node = MCTS::<S,A,E,M,R>::get_or_create_root_node(root, starting_game_state, analytics).await;
        let mut max_depth: usize = 0;

        Self::apply_dirichlet_noise_to_node(&mut root_node, dirichlet, rng);

        while root_node.visits < visits {
            let md = recurse_path_and_expand::<S,A,E,M,R>(root_node, game_engine, analytics, cpuct, temp, rng).await?;

            if md > max_depth {
                max_depth = md;
            }
        }

        let most_visited_action = Self::get_most_visited_action(&root_node, rng)?;

        Ok((most_visited_action, max_depth))
    }

    pub async fn advance_to_action(&mut self, action: A) -> Result<(), &'static str> {
        let mut root = self.root.take();
        let analytics = &mut self.analytics;
        let starting_game_state = &mut self.starting_game_state;
        let game_engine = &self.game_engine;
        let mut root_node = MCTS::<S,A,E,M,R>::get_or_create_root_node(&mut root, starting_game_state, analytics).await;

        let mut node = Self::take_node_of_action(&mut root_node, &action)?;

        if node.is_none() {
            node.replace(Self::expand_leaf(&root_node.game_state, &action, game_engine, analytics).await.0);
        }

        self.root.replace(node.unwrap());

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

    fn select_path_using_PUCT(nodes: &'a mut Vec<MCTSChildNode<S, A>>, Nsb: usize, game_state: &S, cpuct: f64, temp: f64, rng: &mut R) -> Result<&'a mut MCTSChildNode<S, A>, &'static str> {
        let mut pucts = Self::get_PUCT_for_nodes(nodes, Nsb, game_state, cpuct);

        // let temp = temp(game_state);
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

    fn get_PUCT_for_nodes(nodes: &'a mut Vec<MCTSChildNode<S, A>>, Nsb: usize, game_state: &S, cpuct: f64) -> Vec<NodePUCT<'a, S, A>> {
        nodes.iter_mut().map(|child| {
            let child_node = &child.node;
            let Psa = child.policy_score;
            let Nsa = child_node.as_ref().map_or(0, |n| { n.visits });

            // let cpuct = cpuct(&game_state, &child.action);
            let root_Nsb = (Nsb as f64).sqrt();
            let Usa = cpuct * Psa * root_Nsb / (1 + Nsa) as f64;

            // Reverse W here since the evaluation of each child node is that from the other player's perspective.
            let Qsa = child_node.as_ref().map_or(0.0, |n| { 1.0 - n.W / n.visits as f64 });
            let PUCT = Qsa + Usa;

            NodePUCT {
                node: child,
                score: PUCT
            }
        }).collect()
    }

    async fn expand_leaf(game_state: &'a S, action: &'a A, game_engine: &'a E, analytics: &'a M) -> (MCTSNode<S, A>, StateAnalysisValue) {
        let new_game_state = game_engine.take_action(game_state, action);
        MCTS::<S,A,E,M,R>::analyse_and_create_node(new_game_state, analytics).await
    }

    async fn get_or_create_root_node(
        root: &'a mut Option<MCTSNode<S, A>>,
        starting_game_state: &'a mut Option<S>,
        analytics: &'a M
    ) -> &'a mut MCTSNode<S, A> {
        let starting_game_state = starting_game_state.take();

        if root.is_none() {
            root.replace(Self::analyse_and_create_node(
                starting_game_state.expect("Tried to use the same starting game state twice"),
                analytics
            ).await.0);
        }

        root.as_mut().unwrap()
    }

    // Value range is [-1, 1] for the "get_state_analysis" method. However internally for the MCTS a range of
    // [0, 1] is used.
    async fn analyse_and_create_node(game_state: S, analytics: &'a M) -> (MCTSNode<S, A>, StateAnalysisValue) {
        let analysis_result = analytics.get_state_analysis(&game_state).await;

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

            let updated_policy_scores = MCTS::<S,A,E,M,R>::apply_dirichlet_noise(policy_scores, dirichlet, rng);

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

#[allow(non_snake_case)]
async fn recurse_path_and_expand<'a,S,A,E,M,R>(node: &'a mut MCTSNode<S, A>, game_engine: &'a E, analytics: &'a M, cpuct: f64, temp: f64, rng: &'a mut R) -> Result<usize, &'static str>
where
    S: GameState,
    A: Clone + Eq + Debug,
    E: GameEngine<State=S,Action=A>,
    M: GameAnalytics<State=S,Action=A>,
    R: Rng
{
    let mut depth = 0;
    let mut Ws_to_update: Vec<&mut f64> = Vec::new();
    let mut node = node;
    let value_score: f64;

    loop {
        let W = node.W;
        let visits = node.visits;
        depth += 1;
        node.visits += 1;
        Ws_to_update.push(&mut node.W);

        // If the node is a terminal node.
        let children = &mut node.children;
        if children.len() == 0 {
            value_score = W / visits as f64;
            break;
        }

        let game_state = &node.game_state;
        let selected_child_node = MCTS::<S,A,E,M,R>::select_path_using_PUCT(children, visits, game_state, cpuct, temp, rng)?;

        if selected_child_node.node.is_none() {
            let (expanded_node, state_analysis) = MCTS::<S,A,E,M,R>::expand_leaf(
                game_state,
                &selected_child_node.action,
                game_engine,
                analytics
            ).await;

            selected_child_node.node.replace(expanded_node);

            // Flip the score in this case because we are going one node deeper and that viewpoint is from
            // the next player and not the current node's player.
            value_score = 1.0 - state_analysis.value_score;
            break;
        }

        let mut_node = selected_child_node.node.as_mut();
        node = mut_node.unwrap();
    }

    // Reverse the value score at each depth according to the player's valuation perspective.
    for (i, W) in Ws_to_update.into_iter().rev().enumerate() {
        let score = if i % 2 == 0 { value_score } else { 1.0 - value_score };
        *W = *W + score;
    }

    Ok(depth)
}



// #[cfg(test)]
// mod tests {
//     use std::task::{Context,Poll};
//     use std::pin::Pin;
//     use std::future::Future;
//     use uuid::Uuid;
//     use super::*;
//     use super::super::rng;
//     use super::super::game_state::{GameState};
//     use super::super::analytics::{GameStateAnalysis};

//     #[derive(Hash, PartialEq, Eq, Clone, Debug)]
//     struct CountingGameState {
//         pub p1_turn: bool,
//         pub count: usize
//     }

//     impl CountingGameState {
//         fn from_starting_count(p1_turn: bool, count: usize) -> Self {
//             Self { p1_turn, count }
//         }

//         fn is_terminal_state(&self) -> Option<f64> {
//             if self.count == 100 {
//                 Some(if self.p1_turn { 1.0 } else { -1.0 })
//             } else if self.count == 0 {
//                 Some(if self.p1_turn { -1.0 } else { 1.0 })
//             } else {
//                 None
//             }
//         }
//     }

//     impl GameState for CountingGameState {
//         fn initial() -> Self {
//             Self { p1_turn: true, count: 50 }
//         }
//     }

//     struct CountingGameEngine {

//     }

//     impl CountingGameEngine {
//         fn new() -> Self { Self {} }
//     }

//     #[derive(PartialEq, Eq, Clone, Debug)]
//     enum CountingAction {
//         Increment,
//         Decrement,
//         Stay
//     }

//     impl GameEngine for CountingGameEngine {
//         type Action = CountingAction;
//         type State = CountingGameState;

//         fn take_action(&self, game_state: &Self::State, action: &Self::Action) -> Self::State {
//             let count = game_state.count;

//             let new_count = match action {
//                 CountingAction::Increment => count + 1,
//                 CountingAction::Decrement => count - 1,
//                 CountingAction::Stay => count
//             };

//             Self::State { p1_turn: !game_state.p1_turn, count: new_count }
//         }

//         fn is_terminal_state(&self, game_state: &Self::State) -> Option<f64> {
//             game_state.is_terminal_state()
//         }
//     }

//     struct CountingAnalytics {

//     }

//     impl CountingAnalytics {
//         fn new() -> Self { Self {} }
//     }

//     struct CountingGameStateAnalysisFuture {
//         value: GameStateAnalysis<CountingAction>
//     }

//     impl CountingGameStateAnalysisFuture {
//         fn new(value: GameStateAnalysis<CountingAction>) -> Self {
//             Self { value }
//         }
//     }

//     impl Future for CountingGameStateAnalysisFuture {
//         type Output = GameStateAnalysis<CountingAction>;

//         fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
//             Poll::Ready(self.value)
//         }
//     }

//     impl GameAnalytics for CountingAnalytics {
//         type Action = CountingAction;
//         type State = CountingGameState;
//         type Future = CountingGameStateAnalysisFuture;

//         fn get_state_analysis(&self, game_state: &Self::State) -> CountingGameStateAnalysisFuture {
//             let count = game_state.count as f64;

//             if let Some(score) = game_state.is_terminal_state() {
//                 return CountingGameStateAnalysisFuture::new(GameStateAnalysis {
//                     policy_scores: Vec::new(),
//                     value_score: score
//                 });
//             }
            
//             CountingGameStateAnalysisFuture::new(GameStateAnalysis {
//                 policy_scores: vec!(
//                     ActionWithPolicy {
//                         action: CountingAction::Increment,
//                         policy_score: 0.3
//                     },
//                     ActionWithPolicy {
//                         action: CountingAction::Decrement,
//                         policy_score: 0.3
//                     },
//                     ActionWithPolicy {
//                         action: CountingAction::Stay,
//                         policy_score: 0.4
//                     },
//                 ),
//                 value_score: (if game_state.p1_turn { count } else { 100.0 - count } / 50.0) - 1.0
//             })
//         }
//     }

//     #[test]
//     fn test_mcts_is_deterministic() {
//         let game_state = CountingGameState::initial();
//         let game_engine = CountingGameEngine::new();
//         let analytics = CountingAnalytics::new();
//         let uuid = Uuid::parse_str("f555a572-67eb-45fe-83a8-ec90eda83b55").unwrap();

//         let mut mcts = MCTS::new(game_state.to_owned(), &game_engine, &analytics, MCTSOptions::new(
//             None,
//             &|_,_| { 1.0 },
//             &|_| { 0.0 },
//             rng::create_rng_from_uuid(uuid)
//         ));

//         let mut mcts2 = MCTS::new(game_state, &game_engine, &analytics, MCTSOptions::new(
//             None,
//             &|_,_| { 1.0 },
//             &|_| { 0.0 },
//             rng::create_rng_from_uuid(uuid)
//         ));

//         mcts.search(800).unwrap();
//         mcts2.search(800).unwrap();

//         let metrics = mcts.get_root_node_metrics();
//         let metrics2 = mcts.get_root_node_metrics();

//         assert_eq!(metrics, metrics2);
//     }

//     #[test]
//     fn test_mcts_chooses_best_p1_move() {
//         let game_state = CountingGameState::initial();
//         let game_engine = CountingGameEngine::new();
//         let analytics = CountingAnalytics::new();
//         let uuid = Uuid::parse_str("f555a572-67eb-45fe-83a8-ec90eda83b55").unwrap();

//         let mut mcts = MCTS::new(game_state, &game_engine, &analytics, MCTSOptions::new(
//             None,
//             &|_,_| { 1.0 },
//             &|_| { 0.0 },
//             rng::create_rng_from_uuid(uuid)
//         ));

//         let (action, _) = mcts.search(800).unwrap();

//         assert_eq!(action, CountingAction::Increment);
//     }

//     #[test]
//     fn test_mcts_chooses_best_p2_move() {
//         let game_state = CountingGameState::initial();
//         let game_engine = CountingGameEngine::new();
//         let analytics = CountingAnalytics::new();
//         let uuid = Uuid::parse_str("f555a572-67eb-45fe-83a8-ec90eda83b55").unwrap();

//         let mut mcts = MCTS::new(game_state, &game_engine, &analytics, MCTSOptions::new(
//             None,
//             &|_,_| { 1.0 },
//             &|_| { 0.0 },
//             rng::create_rng_from_uuid(uuid)
//         ));

//         mcts.advance_to_action(&CountingAction::Increment).unwrap();

//         let (action, _) = mcts.search(800).unwrap();

//         assert_eq!(action, CountingAction::Decrement);
//     }

//     #[test]
//     fn test_mcts_advance_to_next_works_without_search() {
//         let game_state = CountingGameState::initial();
//         let game_engine = CountingGameEngine::new();
//         let analytics = CountingAnalytics::new();
//         let uuid = Uuid::parse_str("f555a572-67eb-45fe-83a8-ec90eda83b55").unwrap();

//         let mut mcts = MCTS::new(game_state, &game_engine, &analytics, MCTSOptions::new(
//             None,
//             &|_,_| { 1.0 },
//             &|_| { 0.0 },
//             rng::create_rng_from_uuid(uuid)
//         ));

//         mcts.advance_to_action(&CountingAction::Increment).unwrap();
//     }

//     #[test]
//     fn test_mcts_metrics_returns_accurate_results() {
//         let game_state = CountingGameState::initial();
//         let game_engine = CountingGameEngine::new();
//         let analytics = CountingAnalytics::new();
//         let uuid = Uuid::parse_str("f555a572-67eb-45fe-83a8-ec90eda83b55").unwrap();

//         let mut mcts = MCTS::new(game_state, &game_engine, &analytics, MCTSOptions::new(
//             None,
//             &|_,_| { 1.0 },
//             &|_| { 0.0 },
//             rng::create_rng_from_uuid(uuid)
//         ));

//         mcts.search(800).unwrap();

//         let metrics = mcts.get_root_node_metrics().unwrap();

//         assert_eq!(metrics, NodeMetrics {
//             visits: 800,
//             W: 400.91999999999973,
//             children_visits: vec!(
//                 (CountingAction::Increment, 316),
//                 (CountingAction::Decrement, 178),
//                 (CountingAction::Stay, 305)
//             )
//         });
//     }

//     #[test]
//     fn test_mcts_weights_policy_initially() {
//         let game_state = CountingGameState::initial();
//         let game_engine = CountingGameEngine::new();
//         let analytics = CountingAnalytics::new();
//         let uuid = Uuid::parse_str("f555a572-67eb-45fe-83a8-ec90eda83b55").unwrap();

//         let mut mcts = MCTS::new(game_state, &game_engine, &analytics, MCTSOptions::new(
//             None,
//             &|_,_| { 1.0 },
//             &|_| { 0.0 },
//             rng::create_rng_from_uuid(uuid)
//         ));

//         mcts.search(100).unwrap();

//         let metrics = mcts.get_root_node_metrics().unwrap();

//         assert_eq!(metrics, NodeMetrics {
//             visits: 100,
//             W: 50.03999999999999,
//             children_visits: vec!(
//                 (CountingAction::Increment, 33),
//                 (CountingAction::Decrement, 27),
//                 (CountingAction::Stay, 39)
//             )
//         });
//     }

//     #[test]
//     fn test_mcts_works_with_single_node() {
//         let game_state = CountingGameState::initial();
//         let game_engine = CountingGameEngine::new();
//         let analytics = CountingAnalytics::new();
//         let uuid = Uuid::parse_str("f555a572-67eb-45fe-83a8-ec90eda83b55").unwrap();

//         let mut mcts = MCTS::new(game_state, &game_engine, &analytics, MCTSOptions::new(
//             None,
//             &|_,_| { 1.0 },
//             &|_| { 0.0 },
//             rng::create_rng_from_uuid(uuid)
//         ));

//         mcts.search(1).unwrap();

//         let metrics = mcts.get_root_node_metrics().unwrap();

//         assert_eq!(metrics, NodeMetrics {
//             visits: 1,
//             W: 0.5,
//             children_visits: vec!(
//                 (CountingAction::Increment, 0),
//                 (CountingAction::Decrement, 0),
//                 (CountingAction::Stay, 0)
//             )
//         });
//     }

//     #[test]
//     fn test_mcts_works_with_two_nodes() {
//         let game_state = CountingGameState::initial();
//         let game_engine = CountingGameEngine::new();
//         let analytics = CountingAnalytics::new();
//         let uuid = Uuid::parse_str("f555a572-67eb-45fe-83a8-ec90eda83b55").unwrap();

//         let mut mcts = MCTS::new(game_state, &game_engine, &analytics, MCTSOptions::new(
//             None,
//             &|_,_| { 1.0 },
//             &|_| { 0.0 },
//             rng::create_rng_from_uuid(uuid)
//         ));

//         mcts.search(2).unwrap();

//         let metrics = mcts.get_root_node_metrics().unwrap();

//         assert_eq!(metrics, NodeMetrics {
//             visits: 2,
//             W: 1.0,
//             children_visits: vec!(
//                 (CountingAction::Increment, 0),
//                 (CountingAction::Decrement, 0),
//                 (CountingAction::Stay, 1)
//             )
//         });
//     }

//     #[test]
//     fn test_mcts_works_from_provided_non_initial_game_state() {
//         let game_state = CountingGameState::from_starting_count(true, 95);
//         let game_engine = CountingGameEngine::new();
//         let analytics = CountingAnalytics::new();
//         let uuid = Uuid::parse_str("f555a572-67eb-45fe-83a8-ec90eda83b55").unwrap();

//         let mut mcts = MCTS::new(game_state, &game_engine, &analytics, MCTSOptions::new(
//             None,
//             &|_,_| { 0.1 },
//             &|_| { 0.0 },
//             rng::create_rng_from_uuid(uuid)
//         ));

//         mcts.search(8000).unwrap();

//         let metrics = mcts.get_root_node_metrics().unwrap();

//         assert_eq!(metrics, NodeMetrics {
//             visits: 8000,
//             W: 6875.169999999976,
//             children_visits: vec!(
//                 (CountingAction::Increment, 6462),
//                 (CountingAction::Decrement, 728),
//                 (CountingAction::Stay, 809)
//             )
//         });
//     }

//     #[test]
//     fn test_mcts_correctly_handles_terminal_nodes_1() {
//         let game_state = CountingGameState::from_starting_count(false, 99);
//         let game_engine = CountingGameEngine::new();
//         let analytics = CountingAnalytics::new();
//         let uuid = Uuid::parse_str("f555a572-67eb-45fe-83a8-ec90eda83b55").unwrap();

//         let mut mcts = MCTS::new(game_state, &game_engine, &analytics, MCTSOptions::new(
//             None,
//             &|_,_| { 1.0 },
//             &|_| { 0.0 },
//             rng::create_rng_from_uuid(uuid)
//         ));

//         mcts.search(800).unwrap();

//         let metrics = mcts.get_root_node_metrics().unwrap();

//         assert_eq!(metrics, NodeMetrics {
//             visits: 800,
//             W: 8.999999999999986,
//             children_visits: vec!(
//                 (CountingAction::Increment, 182),
//                 (CountingAction::Decrement, 314),
//                 (CountingAction::Stay, 303)
//             )
//         });
//     }

//     #[test]
//     fn test_mcts_correctly_handles_terminal_nodes_2() {
//         let game_state = CountingGameState::from_starting_count(true, 98);
//         let game_engine = CountingGameEngine::new();
//         let analytics = CountingAnalytics::new();
//         let uuid = Uuid::parse_str("f555a572-67eb-45fe-83a8-ec90eda83b55").unwrap();

//         let mut mcts = MCTS::new(game_state, &game_engine, &analytics, MCTSOptions::new(
//             None,
//             &|_,_| { 1.0 },
//             &|_| { 0.0 },
//             rng::create_rng_from_uuid(uuid)
//         ));

//         mcts.search(800).unwrap();

//         let metrics = mcts.get_root_node_metrics().unwrap();

//         assert_eq!(metrics, NodeMetrics {
//             visits: 800,
//             W: 784.5800000000062,
//             children_visits: vec!(
//                 (CountingAction::Increment, 314),
//                 (CountingAction::Decrement, 180),
//                 (CountingAction::Stay, 305)
//             )
//         });
//     }
// }
