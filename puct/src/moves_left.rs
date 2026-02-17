use std::{marker::PhantomData, sync::Mutex};

use common::{CPUCT, GameLength, PlayerToMove};
use engine::Value;

use crate::{EdgeInfo, RollupStats, SelectionPolicy, ValueModel, WeightedMerge};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MovesLeftSnapshot {
    pub p1_value: f32,
    pub p2_value: f32,
    pub game_length: f32,
    total_weight: u32,
}

impl MovesLeftSnapshot {
    #[inline]
    pub fn value_for_player(&self, player_to_move: usize) -> f32 {
        match player_to_move {
            1 => self.p1_value,
            2 => self.p2_value,
            _ => self.p1_value,
        }
    }

    #[inline]
    pub fn game_length(&self) -> f32 {
        self.game_length
    }
}

impl WeightedMerge for MovesLeftSnapshot {
    fn zero() -> Self {
        Self {
            p1_value: 0.0,
            p2_value: 0.0,
            game_length: 0.0,
            total_weight: 0,
        }
    }

    fn merge_weighted(&mut self, other: &Self, weight: u32) {
        if weight == 0 {
            return;
        }

        let prev_weight = self.total_weight;
        let new_weight = prev_weight.saturating_add(weight);

        if prev_weight == 0 {
            self.p1_value = other.p1_value;
            self.p2_value = other.p2_value;
            self.game_length = other.game_length;
            self.total_weight = weight;
            return;
        }

        let prev_weight_f = prev_weight as f32;
        let weight_f = weight as f32;
        let new_weight_f = new_weight as f32;

        self.p1_value = (self.p1_value * prev_weight_f + other.p1_value * weight_f) / new_weight_f;
        self.p2_value = (self.p2_value * prev_weight_f + other.p2_value * weight_f) / new_weight_f;
        self.game_length =
            (self.game_length * prev_weight_f + other.game_length * weight_f) / new_weight_f;
        self.total_weight = new_weight;
    }
}

#[derive(Default)]
pub struct MovesLeftRollup {
    mean: Mutex<MovesLeftSnapshot>,
}

impl From<MovesLeftSnapshot> for MovesLeftRollup {
    fn from(value: MovesLeftSnapshot) -> Self {
        Self {
            mean: Mutex::new(value),
        }
    }
}

impl RollupStats for MovesLeftRollup {
    type Snapshot = MovesLeftSnapshot;

    fn snapshot(&self) -> Self::Snapshot {
        *self.mean.lock().expect("poisoned rollup mutex")
    }

    fn set(&self, value: Self::Snapshot) {
        *self.mean.lock().expect("poisoned rollup mutex") = value;
    }
}

#[derive(Default)]
pub struct MovesLeftValueModel<S, P, T> {
    _phantom: PhantomData<(S, P, T)>,
}

impl<S, P, T> MovesLeftValueModel<S, P, T> {
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<S, P, T> ValueModel for MovesLeftValueModel<S, P, T>
where
    P: Value + GameLength,
    T: Value,
{
    type State = S;
    type Predictions = P;
    type Terminal = T;
    type Rollup = MovesLeftRollup;

    fn pred_snapshot(
        &self,
        _game_state: &Self::State,
        predictions: &Self::Predictions,
    ) -> MovesLeftSnapshot {
        MovesLeftSnapshot {
            p1_value: predictions.get_value_for_player(1),
            p2_value: predictions.get_value_for_player(2),
            game_length: predictions.game_length_score(),
            total_weight: 1,
        }
    }

    fn terminal_snapshot(
        &self,
        _state: &Self::State,
        terminal: &Self::Terminal,
    ) -> MovesLeftSnapshot {
        MovesLeftSnapshot {
            p1_value: terminal.get_value_for_player(1),
            p2_value: terminal.get_value_for_player(2),
            // Terminal outcomes generally don't carry a meaningful "expected game length"
            // prediction; keep it neutral.
            game_length: 0.0,
            total_weight: 1,
        }
    }
}

pub struct MovesLeftStrategyOptions {
    pub fpu: f32,
    pub fpu_root: f32,
    pub moves_left_threshold: f32,
    pub moves_left_scale: f32,
    pub moves_left_factor: f32,
}

pub struct MovesLeftSelectionPolicy<C, S> {
    cpuct: C,
    options: MovesLeftStrategyOptions,
    _phantom: PhantomData<S>,
}

impl<C, S> MovesLeftSelectionPolicy<C, S> {
    pub fn new(cpuct: C, options: MovesLeftStrategyOptions) -> Self {
        Self {
            cpuct,
            options,
            _phantom: PhantomData,
        }
    }

    fn game_length_baseline<'a, A>(
        edges: &[EdgeInfo<'a, A, MovesLeftSnapshot>],
        moves_left_threshold: f32,
        player_to_move: usize,
    ) -> GameLengthBaseline
    where
        A: 'a,
    {
        if moves_left_threshold >= 1.0 {
            return GameLengthBaseline::None;
        }

        let mut best: Option<(u32, MovesLeftSnapshot)> = None;
        for edge in edges {
            if edge.visits == 0 {
                continue;
            }
            let Some(snap) = edge.snapshot else {
                continue;
            };
            match best {
                None => best = Some((edge.visits, snap)),
                Some((best_visits, _)) if edge.visits > best_visits => {
                    best = Some((edge.visits, snap))
                }
                Some(_) => {}
            }
        }

        let Some((_v, snap)) = best else {
            return GameLengthBaseline::None;
        };

        let qsa = snap.value_for_player(player_to_move);
        let expected_game_length = snap.game_length();

        if qsa >= moves_left_threshold {
            GameLengthBaseline::MinimizeGameLength(expected_game_length)
        } else if qsa <= (1.0 - moves_left_threshold) {
            GameLengthBaseline::MaximizeGameLength(expected_game_length)
        } else {
            GameLengthBaseline::None
        }
    }

    fn msa(
        snap: Option<MovesLeftSnapshot>,
        baseline: &GameLengthBaseline,
        options: &MovesLeftStrategyOptions,
    ) -> f32 {
        let Some(snap) = snap else {
            return 0.0;
        };

        let (direction, baseline_len) = match baseline {
            GameLengthBaseline::None => return 0.0,
            GameLengthBaseline::MinimizeGameLength(v) => (1.0f32, *v),
            GameLengthBaseline::MaximizeGameLength(v) => (-1.0f32, *v),
        };

        let expected_game_length = snap.game_length();
        let moves_left_scale = options.moves_left_scale;
        let moves_left_clamped = (baseline_len - expected_game_length)
            .min(moves_left_scale)
            .max(-moves_left_scale);
        let moves_left_scaled = moves_left_clamped / moves_left_scale;
        moves_left_scaled * options.moves_left_factor * direction
    }
}

impl<C, S> SelectionPolicy<MovesLeftSnapshot> for MovesLeftSelectionPolicy<C, S>
where
    C: CPUCT<State = S>,
    S: PlayerToMove,
{
    type State = S;

    fn select_edge<'a, I, A: 'a>(
        &self,
        edges: I,
        node_visits: u32,
        state: &Self::State,
        depth: u32,
    ) -> usize
    where
        I: Iterator<Item = EdgeInfo<'a, A, MovesLeftSnapshot>>,
        MovesLeftSnapshot: 'a,
    {
        let is_root = depth == 0;
        let options = &self.options;

        let fpu = if is_root {
            options.fpu_root
        } else {
            options.fpu
        };
        let root_nsb = (node_visits as f32).sqrt();
        let cpuct = self.cpuct.cpuct(state, node_visits, is_root);

        let player_to_move = state.player_to_move();
        let edges: Vec<EdgeInfo<'a, A, MovesLeftSnapshot>> = edges.collect();
        let baseline =
            Self::game_length_baseline(&edges, options.moves_left_threshold, player_to_move);

        let mut best_index = 0usize;
        let mut best_score = f32::MIN;

        for edge in edges {
            let nsa = edge.visits;
            let psa = edge.policy_prior;
            let usa = cpuct * psa * root_nsb / (1.0 + nsa as f32);

            let qsa = edge
                .snapshot
                .map(|s| s.value_for_player(player_to_move))
                .unwrap_or(fpu);

            let msa = Self::msa(edge.snapshot, &baseline, options);
            let score = msa + qsa + usa;
            if score > best_score {
                best_score = score;
                best_index = edge.edge_index;
            }
        }

        best_index
    }
}

enum GameLengthBaseline {
    MinimizeGameLength(f32),
    MaximizeGameLength(f32),
    None,
}

pub fn map_moves_left_to_one_hot(moves_left: f32, moves_left_size: usize) -> Vec<f32> {
    if moves_left_size == 0 {
        return vec![];
    }

    assert!(
        moves_left.is_finite(),
        "Value must be finite (not NaN or infinity)."
    );
    assert!(moves_left >= 0.0, "Value must not be negative.");
    assert!(moves_left <= usize::MAX as f32, "Value must fit in usize.");

    let moves_left = moves_left.round() as usize;
    let moves_left = moves_left.max(1).min(moves_left_size);
    let mut moves_left_one_hot = vec![0f32; moves_left_size];
    moves_left_one_hot[moves_left - 1] = 1.0;

    moves_left_one_hot
}

pub fn moves_left_expected_value<I: Iterator<Item = f32>>(moves_left_scores: I) -> f32 {
    moves_left_scores
        .enumerate()
        .map(|(i, s)| (i + 1) as f32 * s)
        .fold(0.0f32, |s, e| s + e)
}
