use crate::{
    BackpropagationStrategy, EdgeDetails, MCTSEdge, MCTSNode, NodeLendingIterator,
    SelectionStrategy, CPUCT,
};
use anyhow::Result;
use common::{MovesLeftPropagatedValue, PropagatedGameLength, PropagatedValue};
use engine::{GameEngine, Value};

pub struct MovesLeftSelectionStrategy<S, A, P, C> {
    cpuct: C,
    options: MovesLeftStrategyOptions,
    _phantom: std::marker::PhantomData<(S, A, P)>,
}

pub struct MovesLeftStrategyOptions {
    pub fpu: f32,
    pub fpu_root: f32,
    pub temperature_visit_offset: f32,
    pub moves_left_threshold: f32,
    pub moves_left_scale: f32,
    pub moves_left_factor: f32,
}

#[allow(clippy::too_many_arguments)]
impl MovesLeftStrategyOptions {
    pub fn new(
        fpu: f32,
        fpu_root: f32,
        temperature_visit_offset: f32,
        moves_left_threshold: f32,
        moves_left_scale: f32,
        moves_left_factor: f32,
    ) -> Self {
        MovesLeftStrategyOptions {
            fpu,
            fpu_root,
            temperature_visit_offset,
            moves_left_threshold,
            moves_left_scale,
            moves_left_factor,
        }
    }
}

#[allow(non_snake_case)]
impl<S, A, P, C> MovesLeftSelectionStrategy<S, A, P, C> {
    pub fn new(cpuct: C, options: MovesLeftStrategyOptions) -> Self {
        Self {
            cpuct,
            options,
            _phantom: std::marker::PhantomData,
        }
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
                let Qsa = pv.value();
                let expected_game_length = pv.game_length();

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
        options: &MovesLeftStrategyOptions,
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

        // @TODO: Verify correctness here between game_length and moves_left
        let expected_game_length = edge.propagated_values().game_length();
        let moves_left_scale = options.moves_left_scale;
        let moves_left_clamped = (game_length_baseline - expected_game_length)
            .min(moves_left_scale)
            .max(-moves_left_scale);
        let moves_left_scaled = moves_left_clamped / moves_left_scale;
        moves_left_scaled * options.moves_left_factor * direction
    }
}

#[allow(non_snake_case)]
impl<S, A, P, C> SelectionStrategy for MovesLeftSelectionStrategy<S, A, P, C>
where
    C: CPUCT<State = S>,
    A: Clone,
{
    type State = S;
    type Action = A;
    type Predictions = P;
    type PropagatedValues = MovesLeftPropagatedValue;

    fn select_path(
        &self,
        node: &mut MCTSNode<A, P, MovesLeftPropagatedValue>,
        game_state: &S,
        is_root: bool,
    ) -> Result<usize>
    where
        C: CPUCT<State = S>,
    {
        let options = &self.options;

        let fpu = if is_root {
            options.fpu_root
        } else {
            options.fpu
        };
        let Nsb = node.visits();
        let root_Nsb = (Nsb as f32).sqrt();
        let cpuct = self.cpuct.cpuct(game_state, Nsb, is_root);
        let game_length_baseline = &Self::get_game_length_baseline(
            node.iter_visited_edges_and_top_unvisited_edge(),
            options.moves_left_threshold,
        );

        let mut best_child_index = 0;
        let mut best_puct = std::f32::MIN;

        for (i, edge) in node.iter_visited_edges_and_top_unvisited_edge().enumerate() {
            let W = edge.propagated_values().value();
            let Nsa = edge.visits();
            let Psa = edge.policy_score();
            let Usa = cpuct * Psa * root_Nsb / (1 + Nsa) as f32;
            let Qsa = if Nsa == 0 { fpu } else { W };
            let Msa = Self::Msa(edge, game_length_baseline, options);

            let PUCT = Msa + Qsa + Usa;

            if PUCT > best_puct {
                best_puct = PUCT;
                best_child_index = i;
            }
        }

        Ok(best_child_index)
    }

    fn node_details(
        &self,
        node: &mut MCTSNode<A, P, MovesLeftPropagatedValue>,
        game_state: &S,
        is_root: bool,
    ) -> Vec<EdgeDetails<A, MovesLeftPropagatedValue>>
    where
        C: CPUCT<State = S>,
    {
        let options = &self.options;

        let fpu = if is_root {
            options.fpu_root
        } else {
            options.fpu
        };
        let Nsb = node.visits();
        let root_Nsb = (Nsb as f32).sqrt();
        let cpuct = self.cpuct.cpuct(game_state, Nsb, is_root);

        let mut pucts = Vec::with_capacity(node.child_len());

        for edge in node.iter_visited_edges() {
            let action = edge.action().clone();
            let propagated_values = edge.propagated_values().clone();

            let W = propagated_values.value();
            let Nsa = edge.visits();
            let Psa = edge.policy_score();
            let Usa = cpuct * Psa * root_Nsb / (1 + Nsa) as f32;
            let Qsa = if Nsa == 0 { fpu } else { W };

            let puct_score = Qsa + Usa;
            pucts.push(EdgeDetails {
                action,
                puct_score,
                Psa,
                Nsa,
                cpuct,
                Usa,
                propagated_values,
            });
        }

        pucts
    }
}

#[allow(non_snake_case)]
impl<A, PV> EdgeDetails<A, PV>
where
    PV: PropagatedGameLength,
{
    pub fn game_length(&self) -> f32 {
        self.propagated_values.game_length()
    }
}

enum GameLengthBaseline {
    MinimizeGameLength(f32),
    MaximizeGameLength(f32),
    None,
}

pub struct MovesLeftBackpropagationStrategy<'a, E, S, A, P> {
    engine: &'a E,
    _phantom: std::marker::PhantomData<(S, A, P)>,
}

impl<'e, E, S, A, P> MovesLeftBackpropagationStrategy<'e, E, S, A, P> {
    pub fn new(engine: &'e E) -> Self {
        Self {
            engine,
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

            // The number of visits at the time of traversal acts like a pseudo virtual visit implementation.
            // It is technically not completely mathematically correct as incremental average may be performed out of order, but it is a good approximation.
            // In an ideal implementation we would force the backpropagation of nodes to be in the order of traversal when dealing with multiple thread searches.
            let visits = node.edge_visits_at_time_of_traversal;
            let current_value = edge_to_update.propagated_values().value();
            let new_value = current_value + (score - current_value) / (visits + 1) as f32;
            *edge_to_update.propagated_values_mut().value_mut() = new_value;

            let current_game_length = edge_to_update.propagated_values().game_length();
            let new_game_length = current_game_length
                + (estimated_game_length - current_game_length) / (visits + 1) as f32;
            *edge_to_update.propagated_values_mut().game_length_mut() += new_game_length;
        }
    }

    fn node_info(&self, game_state: &Self::State) -> Self::NodeInfo {
        MovesLeftNodeInfo {
            player_to_move: self.engine.player_to_move(game_state),
        }
    }
}
