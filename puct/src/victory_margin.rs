use std::{marker::PhantomData, sync::Mutex};

use common::{CPUCT, PlayerToMove, VictoryMargin};
use engine::Value;

use crate::{EdgeInfo, RollupStats, SelectionPolicy, ValueModel, WeightedMerge};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct VictoryMarginSnapshot {
    pub p1_value: f32,
    pub p2_value: f32,
    pub victory_margin: f32,
    total_weight: u32,
}

impl VictoryMarginSnapshot {
    #[inline]
    pub fn value_for_player(&self, player_to_move: usize) -> f32 {
        match player_to_move {
            1 => self.p1_value,
            2 => self.p2_value,
            _ => self.p1_value,
        }
    }

    #[inline]
    pub fn victory_margin(&self) -> f32 {
        self.victory_margin
    }
}

impl WeightedMerge for VictoryMarginSnapshot {
    fn zero() -> Self {
        Self {
            p1_value: 0.0,
            p2_value: 0.0,
            victory_margin: 0.0,
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
            self.victory_margin = other.victory_margin;
            self.total_weight = weight;
            return;
        }

        let prev_weight_f = prev_weight as f32;
        let weight_f = weight as f32;
        let new_weight_f = new_weight as f32;

        self.p1_value = (self.p1_value * prev_weight_f + other.p1_value * weight_f) / new_weight_f;
        self.p2_value = (self.p2_value * prev_weight_f + other.p2_value * weight_f) / new_weight_f;
        self.victory_margin =
            (self.victory_margin * prev_weight_f + other.victory_margin * weight_f) / new_weight_f;
        self.total_weight = new_weight;
    }
}

#[derive(Default)]
pub struct VictoryMarginRollup {
    mean: Mutex<VictoryMarginSnapshot>,
}

impl From<VictoryMarginSnapshot> for VictoryMarginRollup {
    fn from(value: VictoryMarginSnapshot) -> Self {
        Self {
            mean: Mutex::new(value),
        }
    }
}

impl RollupStats for VictoryMarginRollup {
    type Snapshot = VictoryMarginSnapshot;

    fn snapshot(&self) -> Self::Snapshot {
        *self.mean.lock().expect("poisoned rollup mutex")
    }

    fn set(&self, value: Self::Snapshot) {
        *self.mean.lock().expect("poisoned rollup mutex") = value;
    }
}

#[derive(Default)]
pub struct VictoryMarginValueModel<S, P, T> {
    _phantom: PhantomData<(S, P, T)>,
}

impl<S, P, T> VictoryMarginValueModel<S, P, T> {
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<S, P, T> ValueModel for VictoryMarginValueModel<S, P, T>
where
    P: Value + VictoryMargin,
    T: Value,
{
    type State = S;
    type Predictions = P;
    type Terminal = T;
    type Rollup = VictoryMarginRollup;

    fn pred_snapshot(
        &self,
        _state: &Self::State,
        predictions: &Self::Predictions,
    ) -> VictoryMarginSnapshot {
        VictoryMarginSnapshot {
            p1_value: predictions.get_value_for_player(1),
            p2_value: predictions.get_value_for_player(2),
            victory_margin: predictions.victory_margin_score(),
            total_weight: 1,
        }
    }

    fn terminal_snapshot(&self, _state: &Self::State, terminal: &Self::Terminal) -> VictoryMarginSnapshot {
        VictoryMarginSnapshot {
            p1_value: terminal.get_value_for_player(1),
            p2_value: terminal.get_value_for_player(2),
            victory_margin: 0.0,
            total_weight: 1,
        }
    }
}

pub struct VictoryMarginStrategyOptions {
    pub fpu: f32,
    pub fpu_root: f32,
    pub victory_margin_threshold: f32,
    pub victory_margin_factor: f32,
}

pub struct VictoryMarginSelectionPolicy<C, S> {
    cpuct: C,
    options: VictoryMarginStrategyOptions,
    _phantom: PhantomData<S>,
}

impl<C, S> VictoryMarginSelectionPolicy<C, S> {
    pub fn new(cpuct: C, options: VictoryMarginStrategyOptions) -> Self {
        Self {
            cpuct,
            options,
            _phantom: PhantomData,
        }
    }

    fn victory_margin_directive<'a, A>(
        edges: &[EdgeInfo<'a, A, VictoryMarginSnapshot>],
        victory_margin_threshold: f32,
        player_to_move: usize,
    ) -> VictoryMarginDirective
    where
        A: 'a,
    {
        if victory_margin_threshold >= 1.0 {
            return VictoryMarginDirective::None;
        }

        let mut best: Option<(u32, VictoryMarginSnapshot)> = None;
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
            return VictoryMarginDirective::None;
        };

        let qsa = snap.value_for_player(player_to_move);

        if qsa >= victory_margin_threshold {
            VictoryMarginDirective::MaximizeVictoryMargin
        } else if qsa <= (1.0 - victory_margin_threshold) {
            VictoryMarginDirective::MinimizeVictoryMargin
        } else {
            VictoryMarginDirective::None
        }
    }

    fn vmsa(
        snap: Option<VictoryMarginSnapshot>,
        directive: &VictoryMarginDirective,
        options: &VictoryMarginStrategyOptions,
        edge_visits: u32,
    ) -> f32 {
        if edge_visits == 0 {
            return 0.0;
        }

        let Some(snap) = snap else {
            return 0.0;
        };

        let direction = match directive {
            VictoryMarginDirective::MaximizeVictoryMargin => 1.0f32,
            VictoryMarginDirective::MinimizeVictoryMargin => -1.0f32,
            VictoryMarginDirective::None => return 0.0,
        };

        snap.victory_margin() * options.victory_margin_factor * direction
    }
}

impl<C, S> SelectionPolicy<VictoryMarginSnapshot> for VictoryMarginSelectionPolicy<C, S>
where
    C: CPUCT<State = S>,
    S: PlayerToMove,
{
    type State = S;

    fn select_edge<'a, I, A: 'a>(&self, edges: I, node_visits: u32, state: &Self::State, depth: u32) -> usize
    where
        I: Iterator<Item = EdgeInfo<'a, A, VictoryMarginSnapshot>>,
        VictoryMarginSnapshot: 'a,
    {
        let is_root = depth == 0;
        let options = &self.options;

        let fpu = if is_root { options.fpu_root } else { options.fpu };
        let root_nsb = (node_visits as f32).sqrt();
        let cpuct = self.cpuct.cpuct(state, node_visits, is_root);

        let player_to_move = state.player_to_move();
        let edges: Vec<EdgeInfo<'a, A, VictoryMarginSnapshot>> = edges.collect();
        let directive = Self::victory_margin_directive(
            &edges,
            options.victory_margin_threshold,
            player_to_move,
        );

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

            let vmsa = Self::vmsa(edge.snapshot, &directive, options, nsa);
            let score = vmsa + qsa + usa;
            if score > best_score {
                best_score = score;
                best_index = edge.edge_index;
            }
        }

        best_index
    }
}

enum VictoryMarginDirective {
    MinimizeVictoryMargin,
    MaximizeVictoryMargin,
    None,
}
