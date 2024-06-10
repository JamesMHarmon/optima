use crate::{MCTSEdge, MCTSNode, MCTSOptions, CPUCT};
use anyhow::{anyhow, Context, Result};

pub struct MovesLeftSelectionStrategy<S, A, P, PV, C> {
    cpuct: C,
    _phantom: std::marker::PhantomData<(S, A, P, PV)>,
}

enum GameLengthBaseline {
    MinimizeGameLength(f32),
    MaximizeGameLength(f32),
    None,
}

#[allow(non_snake_case)]
impl<S, A, P, PV, C> MovesLeftSelectionStrategy<S, A, P, PV, C> {
    fn select_path(
        node: &mut MCTSNode<A, P, PV>,
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
        let cpuct = cpuct.cpuct(game_state, Nsb, is_root);
        let game_length_baseline = &Self::get_game_length_baseline(
            node.iter_visited_edges_and_top_unvisited_edge(),
            options.moves_left_threshold,
        );

        let mut best_child_index = 0;
        let mut best_puct = std::f32::MIN;

        for (i, child) in node.iter_visited_edges_and_top_unvisited_edge().enumerate() {
            let W = child.W();
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
        I: Iterator<Item = &'b MCTSEdge<A, PV>>,
        A: 'b,
        PV: 'b,
    {
        if moves_left_threshold >= 1.0 {
            return GameLengthBaseline::None;
        }

        edges
            .max_by_key(|n| n.visits())
            .filter(|n| n.visits() > 0)
            .map_or(GameLengthBaseline::None, |n| {
                let Qsa = n.W() / n.visits() as f32;
                let expected_game_length = n.M() / n.visits() as f32;

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
        child: &MCTSEdge<A, PV>,
        game_length_baseline: &GameLengthBaseline,
        options: &MCTSOptions,
    ) -> f32 {
        if child.visits() == 0 {
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

        let expected_game_length = child.M() / child.visits() as f32;
        let moves_left_scale = options.moves_left_scale;
        let moves_left_clamped = (game_length_baseline - expected_game_length)
            .min(moves_left_scale)
            .max(-moves_left_scale);
        let moves_left_scaled = moves_left_clamped / moves_left_scale;
        moves_left_scaled * options.moves_left_factor * direction
    }
}
