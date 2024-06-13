use crate::{
    BackpropagationStrategy, EdgeDetails, MCTSEdge, MCTSNode, MCTSOptions, NodeLendingIterator,
    CPUCT,
};
use anyhow::Result;
use common::div_or_zero;
use engine::{GameEngine, Value};
use itertools::Itertools;

pub struct MovesLeftSelectionStrategy<S, A, C> {
    cpuct: C,
    _phantom: std::marker::PhantomData<(S, A)>,
}

pub struct MovesLeftPrediction {
    moves_left: f32,
    value: f32,
}

#[allow(non_snake_case)]
impl MovesLeftPrediction {
    pub fn new(moves_left: f32, value: f32) -> Self {
        Self { moves_left, value }
    }
}

#[derive(Default, Clone)]
pub struct MovesLeftPropagatedValue {
    game_length: f32,
    value: f32,
}

enum GameLengthBaseline {
    MinimizeGameLength(f32),
    MaximizeGameLength(f32),
    None,
}

#[allow(non_snake_case)]
impl<S, A, C> MovesLeftSelectionStrategy<S, A, C> {
    fn select_path(
        &self,
        node: &mut MCTSNode<A, MovesLeftPrediction, MovesLeftPropagatedValue>,
        game_state: &S,
        is_root: bool,
        options: &MCTSOptions,
    ) -> Result<usize>
    where
        C: CPUCT<State = S>,
    {
        let fpu = if is_root {
            options.fpu_root
        } else {
            options.fpu
        };
        let Nsb = node.get_node_visits();
        let root_Nsb = (Nsb as f32).sqrt();
        let cpuct = self.cpuct.cpuct(game_state, Nsb, is_root);
        let game_length_baseline = &Self::get_game_length_baseline(
            node.iter_visited_edges_and_top_unvisited_edge(),
            options.moves_left_threshold,
        );

        let mut best_child_index = 0;
        let mut best_puct = std::f32::MIN;

        for (i, child) in node.iter_visited_edges_and_top_unvisited_edge().enumerate() {
            let W = child.propagated_values().value;
            let Nsa = child.visits();
            let Psa = child.policy_score();
            let Usa = cpuct * Psa * root_Nsb / (1 + Nsa) as f32;
            let Qsa = if Nsa == 0 { fpu } else { W / Nsa as f32 };
            let Msa = Self::Msa(child, game_length_baseline, options);

            let PUCT = Msa + Qsa + Usa;

            if PUCT > best_puct {
                best_puct = PUCT;
                best_child_index = i;
            }
        }

        Ok(best_child_index)
    }

    fn get_game_length_baseline<'b, I>(edges: I, moves_left_threshold: f32) -> GameLengthBaseline
    where
        I: Iterator<Item = &'b MCTSEdge<A, MovesLeftPropagatedValue>>,
        A: 'b,
    {
        if moves_left_threshold >= 1.0 {
            return GameLengthBaseline::None;
        }

        edges
            .max_by_key(|n| n.visits())
            .filter(|n| n.visits() > 0)
            .map_or(GameLengthBaseline::None, |e| {
                let pv = e.propagated_values();
                let Qsa = pv.value / e.visits() as f32;
                let expected_game_length = pv.game_length / e.visits() as f32;

                if Qsa >= moves_left_threshold {
                    GameLengthBaseline::MinimizeGameLength(expected_game_length)
                } else if Qsa <= (1.0 - moves_left_threshold) {
                    GameLengthBaseline::MaximizeGameLength(expected_game_length)
                } else {
                    GameLengthBaseline::None
                }
            })
    }

    fn Msa(
        edge: &MCTSEdge<A, MovesLeftPropagatedValue>,
        game_length_baseline: &GameLengthBaseline,
        options: &MCTSOptions,
    ) -> f32 {
        if edge.visits() == 0 {
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

        let expected_game_length = edge.propagated_values().game_length / edge.visits() as f32;
        let moves_left_scale = options.moves_left_scale;
        let moves_left_clamped = (game_length_baseline - expected_game_length)
            .min(moves_left_scale)
            .max(-moves_left_scale);
        let moves_left_scaled = moves_left_clamped / moves_left_scale;
        moves_left_scaled * options.moves_left_factor * direction
    }

    fn node_details(
        &self,
        node: &mut MCTSNode<A, MovesLeftPrediction, MovesLeftPropagatedValue>,
        game_state: &S,
        is_root: bool,
        options: &MCTSOptions,
    ) -> Vec<EdgeDetails<A, MovesLeftPropagatedValue>>
    where
        C: CPUCT<State = S>,
        A: Clone,
    {
        let fpu = if is_root {
            options.fpu_root
        } else {
            options.fpu
        };
        let Nsb = node.get_node_visits();
        let root_Nsb = (Nsb as f32).sqrt();
        let cpuct = self.cpuct.cpuct(game_state, Nsb, is_root);
        let moves_left_threshold = options.moves_left_threshold;
        let iter_all_edges = node.iter_all_edges().map(|e| &*e);
        let game_length_baseline =
            Self::get_game_length_baseline(iter_all_edges, moves_left_threshold);

        let mut pucts = Vec::with_capacity(node.child_len());

        for edge in node.iter_visited_edges() {
            let action = edge.action().clone();
            let propagated_values = edge.propagated_values().clone();
            let W = propagated_values.value;
            let Nsa = edge.visits();
            let Psa = edge.policy_score();
            let Usa = cpuct * Psa * root_Nsb / (1 + Nsa) as f32;
            let Qsa = if Nsa == 0 { fpu } else { W / Nsa as f32 };
            let Msa = Self::Msa(edge, &game_length_baseline, options);
            let game_length = div_or_zero(propagated_values.game_length, Nsa as f32);

            let puct_score = Qsa + Usa;
            pucts.push(EdgeDetails {
                action,
                puct_score,
                Psa,
                Nsa,
                cpuct,
                Usa,
                game_length,
                propagated_values,
            });
        }

        pucts
    }
}

pub struct MovesLeftBackpropagationStrategy<'a, E, S, A, P> {
    engine: &'a E,
    game_length_baseline: GameLengthBaseline,
    _phantom: std::marker::PhantomData<(S, A, P)>,
}

impl<'e, E, S, A, P> MovesLeftBackpropagationStrategy<'e, E, S, A, P> {
    pub fn new(engine: &'e E, game_length_baseline: GameLengthBaseline) -> Self {
        Self {
            engine,
            game_length_baseline,
            _phantom: std::marker::PhantomData,
        }
    }
}

pub struct MovesLeftNodeInfo {
    pub player_to_move: usize,
}

pub trait GameLength {
    fn game_length_score(&self) -> f32;
}

impl<'a, E, S, A, P> BackpropagationStrategy for MovesLeftBackpropagationStrategy<'a, E, S, A, P>
where
    P: Value + GameLength,
    E: GameEngine<State = S, Action = A>,
{
    type State = S;
    type Action = A;
    type Predictions = P;
    type PropagatedValues = MovesLeftPropagatedValue;
    type NodeInfo = MovesLeftNodeInfo;

    fn backpropagate<'node, I>(&self, visited_nodes: I, predictions: &Self::Predictions)
    where
        I: NodeLendingIterator<
            'node,
            Self::NodeInfo,
            Self::Action,
            Self::Predictions,
            Self::PropagatedValues,
        >,
    {
        let mut visited_nodes = visited_nodes;
        let estimated_game_length = predictions.game_length_score();

        while let Some(node) = visited_nodes.next() {
            // Update value of W from the parent node's perspective.
            // This is because the parent chooses which child node to select, and as such will want the one with the
            // highest V from its perspective. A child node (or edge) does not care what its value (W or Q) is from its own perspective.
            let score = predictions.get_value_for_player(node.node_info.player_to_move);

            let edge_to_update = node.node.get_edge_by_index_mut(node.selected_edge_index);

            edge_to_update.propagated_values_mut().value += score;
            edge_to_update.propagated_values_mut().game_length += estimated_game_length;
        }
    }

    fn node_info(&self, game_state: &Self::State) -> Self::NodeInfo {
        MovesLeftNodeInfo {
            player_to_move: self.engine.player_to_move(game_state),
        }
    }
}
