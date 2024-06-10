use crate::{BackpropagationStrategy, MCTSEdge, MCTSNode, MCTSOptions, CPUCT};
use anyhow::Result;
use generational_arena::Index;

pub struct MovesLeftSelectionStrategy<S, A, C> {
    cpuct: C,
    _phantom: std::marker::PhantomData<(S, A)>,
}

pub struct MovesLeftPrediction {
    moves_left: f32,
    value: f32
}

#[allow(non_snake_case)]
impl MovesLeftPrediction {
    pub fn new(moves_left: f32, value: f32) -> Self {
        Self {
            moves_left,
            value
        }
    }
}

#[derive(Default)]
pub struct MovesLeftPropagatedValue {
    game_length: f32,
    value: f32
}

#[allow(non_snake_case)]
impl MovesLeftPropagatedValue {
    pub fn new(game_length: f32, value: f32) -> Self {
        Self {
            game_length,
            value
        }
    }

    pub fn M(&self) -> f32 {
        self.game_length
    }

    pub fn W(&self) -> f32 {
        self.value
    }
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
        C: CPUCT<State = S>
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
            let W = child.propagated_values().W();
            let Nsa = child.visits();
            let Psa = child.policy_score();
            let Usa = cpuct * Psa * root_Nsb / (1 + Nsa) as f32;
            let Qsa = if Nsa == 0 { fpu } else { W / Nsa as f32 };
            let Msa = Self::get_Msa(child, game_length_baseline, options);

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
        A: 'b
    {
        if moves_left_threshold >= 1.0 {
            return GameLengthBaseline::None;
        }

        edges
            .max_by_key(|n| n.visits())
            .filter(|n| n.visits() > 0)
            .map_or(GameLengthBaseline::None, |e| {
                let pv = e.propagated_values();
                let Qsa = pv.W() / e.visits() as f32;
                let expected_game_length = pv.M() / e.visits() as f32;

                if Qsa >= moves_left_threshold {
                    GameLengthBaseline::MinimizeGameLength(expected_game_length)
                } else if Qsa <= (1.0 - moves_left_threshold) {
                    GameLengthBaseline::MaximizeGameLength(expected_game_length)
                } else {
                    GameLengthBaseline::None
                }
            })
    }

    fn get_Msa(
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

        let expected_game_length = edge.propagated_values().M() / edge.visits() as f32;
        let moves_left_scale = options.moves_left_scale;
        let moves_left_clamped = (game_length_baseline - expected_game_length)
            .min(moves_left_scale)
            .max(-moves_left_scale);
        let moves_left_scaled = moves_left_clamped / moves_left_scale;
        moves_left_scaled * options.moves_left_factor * direction
    }
}

pub struct MovesLeftBackpropagationStrategy {
    game_length_baseline: GameLengthBaseline,
}

impl MovesLeftBackpropagationStrategy {
    pub fn new(game_length_baseline: GameLengthBaseline) -> Self {
        Self {
            game_length_baseline
        }
    }
}

impl<S, A, P, PV> BackpropagationStrategy for MovesLeftBackpropagationStrategy
{
    type State = S;
    type Action = A;
    type Predictions;
    type PredicationValues;
    type NodeInfo;

    fn backpropagate(
        &self,
        visited_nodes: &[NodeUpdateInfo],
        evaluated_node_index: Index,
        evaluated_node_move_num: usize,
        arena: &mut NodeArenaInner<MCTSNode<A, P, PV>>,
    ) {
        let evaluated_node = &arena.node(evaluated_node_index);
        let value_score = &evaluated_node.value_score().clone();
        let evaluated_node_moves_left_score = evaluated_node.moves_left_score();
        let evaluated_node_game_length = evaluated_node_move_num as f32 + evaluated_node_moves_left_score;

        for NodeUpdateInfo {
            parent_node_index,
            child_edge_index,
            parent_node_player_to_move,
        } in visited_nodes
        {
            let node_to_update_parent = arena.node_mut(*parent_node_index);
            // Update value of W from the parent node's perspective.
            // This is because the parent chooses which child node to select, and as such will want the one with the
            // highest V from its perspective. A child node (or edge) does not care what its value (W or Q) is from its own perspective.
            let score = value_score.get_value_for_player(*parent_node_player_to_move);

            let node_to_update = node_to_update_parent.get_child_by_index_mut(*child_edge_index);
            
            node_to_update.add_W(score);
            node_to_update.add_M(evaluated_node_game_length);
        }
    }

    
    fn node_info(&self, game_state: &Self::State) -> Self::NodeInfo {
        todo!()
    }
}
