use anyhow::Result;
use common::PropagatedValue;
use engine::{GameEngine, Value};
use mcts::{
    BackpropagationStrategy, CPUCT, EdgeDetails, MCTSEdge, MCTSNode, NodeLendingIterator,
    SelectionStrategy,
};

use crate::QuoridorPropagatedValue;

pub struct QuoridorSelectionStrategy<S, A, P, C> {
    cpuct: C,
    options: QuoridorStrategyOptions,
    _phantom: std::marker::PhantomData<(S, A, P)>,
}

pub struct QuoridorStrategyOptions {
    pub fpu: f32,
    pub fpu_root: f32,
    pub victory_margin_threshold: f32,
    pub victory_margin_factor: f32,
}

#[allow(clippy::too_many_arguments)]
impl QuoridorStrategyOptions {
    pub fn new(
        fpu: f32,
        fpu_root: f32,
        victory_margin_threshold: f32,
        victory_margin_factor: f32,
    ) -> Self {
        QuoridorStrategyOptions {
            fpu,
            fpu_root,
            victory_margin_threshold,
            victory_margin_factor,
        }
    }
}

#[allow(non_snake_case)]
impl<S, A, P, C> QuoridorSelectionStrategy<S, A, P, C> {
    pub fn new(cpuct: C, options: QuoridorStrategyOptions) -> Self {
        Self {
            cpuct,
            options,
            _phantom: std::marker::PhantomData,
        }
    }

    fn get_victory_margin_baseline<'b, I>(
        edges: I,
        victory_margin_threshold: f32,
    ) -> VictoryMarginDirective
    where
        I: Iterator<Item = &'b MCTSEdge<A, QuoridorPropagatedValue>>,
        A: 'b,
    {
        if victory_margin_threshold >= 1.0 {
            return VictoryMarginDirective::None;
        }

        edges
            .max_by_key(|n| n.visits())
            .filter(|n| n.visits() > 0)
            .map_or(VictoryMarginDirective::None, |e| {
                let pv = e.propagated_values();
                let Qsa = pv.value();
                if Qsa >= victory_margin_threshold {
                    VictoryMarginDirective::MaximizeVictoryMargin
                } else if Qsa <= (1.0 - victory_margin_threshold) {
                    VictoryMarginDirective::MinimizeVictoryMargin
                } else {
                    VictoryMarginDirective::None
                }
            })
    }

    fn VMsa(
        edge: &MCTSEdge<A, QuoridorPropagatedValue>,
        victory_margin_baseline: &VictoryMarginDirective,
        options: &QuoridorStrategyOptions,
    ) -> f32 {
        if edge.visits() == 0 {
            return 0.0;
        }

        if let VictoryMarginDirective::None = victory_margin_baseline {
            return 0.0;
        }

        let direction = match victory_margin_baseline {
            VictoryMarginDirective::MaximizeVictoryMargin => 1.0,
            VictoryMarginDirective::MinimizeVictoryMargin => -1.0,
            _ => 0.0,
        };

        let victory_margin = edge.propagated_values().victory_margin();
        victory_margin * options.victory_margin_factor * direction
    }
}

#[allow(non_snake_case)]
impl<S, A, P, C> SelectionStrategy for QuoridorSelectionStrategy<S, A, P, C>
where
    C: CPUCT<State = S>,
    A: Clone,
{
    type State = S;
    type Action = A;
    type Predictions = P;
    type PropagatedValues = QuoridorPropagatedValue;

    fn select_path(
        &self,
        node: &mut MCTSNode<A, P, QuoridorPropagatedValue>,
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
        let victory_margin_baseline = &Self::get_victory_margin_baseline(
            node.iter_visited_edges_and_top_unvisited_edge(),
            options.victory_margin_threshold,
        );

        let mut best_child_index = 0;
        let mut best_puct = f32::MIN;

        for (i, edge) in node.iter_visited_edges_and_top_unvisited_edge().enumerate() {
            let W = edge.propagated_values().value();
            let Nsa = edge.visits();
            let Psa = edge.policy_score();
            let Usa = cpuct * Psa * root_Nsb / (1 + Nsa) as f32;
            let Qsa = if Nsa == 0 { fpu } else { W };
            let VMsa = Self::VMsa(edge, victory_margin_baseline, options);

            let PUCT = VMsa + Qsa + Usa;

            if PUCT > best_puct {
                best_puct = PUCT;
                best_child_index = i;
            }
        }

        Ok(best_child_index)
    }

    fn node_details(
        &self,
        node: &mut MCTSNode<A, P, QuoridorPropagatedValue>,
        game_state: &S,
        is_root: bool,
    ) -> Vec<EdgeDetails<A, QuoridorPropagatedValue>>
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

enum VictoryMarginDirective {
    MinimizeVictoryMargin,
    MaximizeVictoryMargin,
    None,
}

pub struct QuoridorBackpropagationStrategy<'a, E, S, A, P> {
    engine: &'a E,
    _phantom: std::marker::PhantomData<(S, A, P)>,
}

impl<'e, E, S, A, P> QuoridorBackpropagationStrategy<'e, E, S, A, P> {
    pub fn new(engine: &'e E) -> Self {
        Self {
            engine,
            _phantom: std::marker::PhantomData,
        }
    }
}

pub struct VictoryMarginNodeInfo {
    pub player_to_move: usize,
}

pub trait VictoryMargin {
    fn victory_margin_score(&self) -> f32;
}

impl<E, S, A, P> BackpropagationStrategy for QuoridorBackpropagationStrategy<'_, E, S, A, P>
where
    P: Value + VictoryMargin,
    E: GameEngine<State = S, Action = A>,
{
    type State = S;
    type Action = A;
    type Predictions = P;
    type PropagatedValues = QuoridorPropagatedValue;
    type NodeInfo = VictoryMarginNodeInfo;

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
        let victory_margin_score = predictions.victory_margin_score();

        while let Some(node) = visited_nodes.next() {
            // Update value of W from the parent node's perspective.
            // This is because the parent chooses which child node to select, and as such will want the one with the
            // highest V from its perspective. A child node (or edge) does not care what its value (W or Q) is from its own perspective.
            let value_score = predictions.get_value_for_player(node.node_info.player_to_move);
            let edge_to_update = node.node.get_edge_by_index_mut(node.selected_edge_index);

            let propagated_values = edge_to_update.propagated_values_mut();
            let num_updates = propagated_values.num_updates();

            let value = propagated_values.value();
            let new_value = value + (value_score - value) / (num_updates + 1) as f32;
            *propagated_values.value_mut() = new_value;

            let victory_margin = propagated_values.victory_margin();
            let new_victory_margin =
                victory_margin + (victory_margin_score - victory_margin) / (num_updates + 1) as f32;
            *propagated_values.victory_margin_mut() = new_victory_margin;

            propagated_values.increment_num_updates();
        }
    }

    fn node_info(&self, game_state: &Self::State) -> Self::NodeInfo {
        VictoryMarginNodeInfo {
            player_to_move: self.engine.player_to_move(game_state),
        }
    }
}
