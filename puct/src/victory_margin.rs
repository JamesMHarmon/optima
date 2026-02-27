use core::panic;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use common::{CPUCT, GameLength, PlayerToMove, PlayerValue, VictoryMargin};
use serde::{Deserialize, Serialize};

use crate::{EdgeInfo, NodeInfo, RollupStats, SelectionPolicy, ValueModel, WeightedMerge};

#[derive(Clone, Copy)]
pub struct VictoryMarginStrategyOptions {
    pub fpu: f32,
    pub fpu_root: f32,
    pub victory_margin_threshold: f32,
    pub victory_margin_factor: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct VictoryMarginSnapshot {
    pub p1_sum: f64,
    pub p2_sum: f64,
    pub victory_margin_sum: f64,
    pub game_length_sum: f64,
    pub total_weight: u32,
}

impl Default for VictoryMarginSnapshot {
    fn default() -> Self {
        Self {
            p1_sum: 0.0,
            p2_sum: 0.0,
            victory_margin_sum: 0.0,
            game_length_sum: 0.0,
            total_weight: 0,
        }
    }
}

impl PlayerValue for VictoryMarginSnapshot {
    #[inline]
    fn player_value(&self, player: usize) -> f32 {
        if self.total_weight == 0 {
            return 0.0;
        }

        let denom = self.total_weight as f64;
        match player {
            1 => (self.p1_sum / denom) as f32,
            2 => (self.p2_sum / denom) as f32,
            _ => panic!("Invalid player index: {}", player),
        }
    }
}

impl GameLength for VictoryMarginSnapshot {
    #[inline]
    fn game_length(&self) -> f32 {
        if self.total_weight == 0 {
            return 0.0;
        }
        (self.game_length_sum / (self.total_weight as f64)) as f32
    }
}

impl VictoryMargin for VictoryMarginSnapshot {
    #[inline]
    fn victory_margin(&self) -> f32 {
        if self.total_weight == 0 {
            return 0.0;
        }
        (self.victory_margin_sum / (self.total_weight as f64)) as f32
    }
}

impl WeightedMerge for VictoryMarginSnapshot {
    #[inline]
    fn zero() -> Self {
        Self::default()
    }

    #[inline]
    fn merge_weighted(&mut self, other: &Self, weight: u32) {
        if weight == 0 || other.total_weight == 0 {
            return;
        }

        let incoming_mass = weight as f64;
        let mass_scale = incoming_mass / (other.total_weight as f64);

        self.p1_sum += mass_scale * other.p1_sum;
        self.p2_sum += mass_scale * other.p2_sum;
        self.game_length_sum += mass_scale * other.game_length_sum;
        self.victory_margin_sum += mass_scale * other.victory_margin_sum;

        self.total_weight += weight;
    }
}

pub struct VictoryMarginRollup {
    p1_sum_bits: AtomicU64,
    p2_sum_bits: AtomicU64,
    game_length_sum_bits: AtomicU64,
    vm_sum_bits: AtomicU64,
    total_weight: AtomicU32,
}

impl Default for VictoryMarginRollup {
    fn default() -> Self {
        Self {
            p1_sum_bits: AtomicU64::new(0),
            p2_sum_bits: AtomicU64::new(0),
            game_length_sum_bits: AtomicU64::new(0),
            vm_sum_bits: AtomicU64::new(0),
            total_weight: AtomicU32::new(0),
        }
    }
}

impl VictoryMarginRollup {
    #[inline]
    fn load_f64(a: &AtomicU64) -> f64 {
        f64::from_bits(a.load(Ordering::Relaxed))
    }

    #[inline]
    fn store_f64(a: &AtomicU64, v: f64) {
        a.store(v.to_bits(), Ordering::Relaxed);
    }
}

impl From<VictoryMarginSnapshot> for VictoryMarginRollup {
    fn from(value: VictoryMarginSnapshot) -> Self {
        let rollup = Self::default();
        rollup.set(value);
        rollup
    }
}

impl RollupStats for VictoryMarginRollup {
    type Snapshot = VictoryMarginSnapshot;

    #[inline]
    fn snapshot(&self) -> Self::Snapshot {
        VictoryMarginSnapshot {
            p1_sum: Self::load_f64(&self.p1_sum_bits),
            p2_sum: Self::load_f64(&self.p2_sum_bits),
            victory_margin_sum: Self::load_f64(&self.vm_sum_bits),
            game_length_sum: Self::load_f64(&self.game_length_sum_bits),
            total_weight: self.total_weight.load(Ordering::Relaxed),
        }
    }

    #[inline]
    fn set(&self, value: Self::Snapshot) {
        Self::store_f64(&self.p1_sum_bits, value.p1_sum);
        Self::store_f64(&self.p2_sum_bits, value.p2_sum);
        Self::store_f64(&self.vm_sum_bits, value.victory_margin_sum);
        Self::store_f64(&self.game_length_sum_bits, value.game_length_sum);
        self.total_weight
            .store(value.total_weight, Ordering::Relaxed);
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
    P: PlayerValue + VictoryMargin + GameLength,
    T: PlayerValue + VictoryMargin + GameLength,
{
    type State = S;
    type Predictions = P;
    type Terminal = T;
    type Snapshot = VictoryMarginSnapshot;
    type Rollup = VictoryMarginRollup;

    fn pred_snapshot(
        &self,
        _state: &Self::State,
        predictions: &Self::Predictions,
    ) -> VictoryMarginSnapshot {
        VictoryMarginSnapshot {
            p1_sum: predictions.player_value(1) as f64,
            p2_sum: predictions.player_value(2) as f64,
            victory_margin_sum: predictions.victory_margin() as f64,
            game_length_sum: predictions.game_length() as f64,
            total_weight: 1,
        }
    }

    fn terminal_snapshot(
        &self,
        _state: &Self::State,
        terminal: &Self::Terminal,
    ) -> VictoryMarginSnapshot {
        VictoryMarginSnapshot {
            p1_sum: terminal.player_value(1) as f64,
            p2_sum: terminal.player_value(2) as f64,
            victory_margin_sum: terminal.victory_margin() as f64,
            game_length_sum: terminal.game_length() as f64,
            total_weight: 1,
        }
    }
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

    fn directive_from_baseline(
        baseline: Option<VictoryMarginSnapshot>,
        threshold: f32,
        player_index: usize,
    ) -> VictoryMarginDirective {
        if threshold >= 1.0 {
            return VictoryMarginDirective::None;
        }

        let Some(baseline) = baseline else {
            return VictoryMarginDirective::None;
        };

        let qsa = baseline.player_value(player_index);

        if qsa >= threshold {
            VictoryMarginDirective::MaximizeVictoryMargin
        } else if qsa <= (1.0 - threshold) {
            VictoryMarginDirective::MinimizeVictoryMargin
        } else {
            VictoryMarginDirective::None
        }
    }
}

impl<C, S> SelectionPolicy<VictoryMarginSnapshot> for VictoryMarginSelectionPolicy<C, S>
where
    C: CPUCT<State = S>,
    S: PlayerToMove,
{
    type State = S;

    fn select_edge<'a, I, A: 'a>(&self, node: NodeInfo, edges: I, state: &Self::State) -> usize
    where
        I: Iterator<Item = EdgeInfo<'a, A, VictoryMarginSnapshot>>,
        VictoryMarginSnapshot: 'a,
    {
        let is_root = node.is_root();
        let options = &self.options;

        let fpu = if is_root {
            options.fpu_root
        } else {
            options.fpu
        };

        let node_total_visits = node.total_visits();
        let cpuct = self.cpuct.cpuct(state, node_total_visits, is_root);
        let root_sqrt = (node_total_visits as f32).sqrt();
        let player_index = state.player_to_move(); // 1 or 2

        let mut baseline_visits = 0u32;
        let mut baseline_snap: Option<VictoryMarginSnapshot> = None;

        let mut best_none_index = 0usize;
        let mut best_none_score = f32::MIN;

        let mut best_max_index = 0usize;
        let mut best_max_score = f32::MIN;

        let mut best_min_index = 0usize;
        let mut best_min_score = f32::MIN;

        for edge in edges {
            let visits = edge.visits;
            let virtual_visits = edge.virtual_visits;
            let nsa = visits + virtual_visits;
            let psa = edge.policy_prior;

            let usa = cpuct * psa * root_sqrt / (1.0 + nsa as f32);

            let qsa = edge
                .snapshot
                .map(|s| s.player_value(player_index))
                .unwrap_or(fpu);

            // Virtual loss: treat in-flight (virtual) visits as a loss of 0.0 by down-weighting
            // the Q term according to the fraction of completed visits.
            let qsa = if nsa == 0 {
                qsa
            } else {
                qsa * (visits as f32) / (nsa as f32)
            };

            let base_score = qsa + usa;
            if base_score > best_none_score {
                best_none_score = base_score;
                best_none_index = edge.edge_index;
            }

            let vm_adj = if nsa == 0 {
                0.0
            } else {
                edge.snapshot
                    .map(|s| s.victory_margin() * options.victory_margin_factor)
                    .unwrap_or(0.0)
            };

            let score_max = base_score + vm_adj;
            if score_max > best_max_score {
                best_max_score = score_max;
                best_max_index = edge.edge_index;
            }

            let score_min = base_score - vm_adj;
            if score_min > best_min_score {
                best_min_score = score_min;
                best_min_index = edge.edge_index;
            }

            if nsa > 0
                && let Some(snap) = edge.snapshot
                && nsa > baseline_visits
            {
                baseline_visits = nsa;
                baseline_snap = Some(snap);
            }
        }

        match Self::directive_from_baseline(
            baseline_snap,
            options.victory_margin_threshold,
            player_index,
        ) {
            VictoryMarginDirective::None => best_none_index,
            VictoryMarginDirective::MaximizeVictoryMargin => best_max_index,
            VictoryMarginDirective::MinimizeVictoryMargin => best_min_index,
        }
    }
}

impl Eq for VictoryMarginSnapshot {}

impl std::fmt::Display for VictoryMarginSnapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "p1: {:.3}, p2: {:.3}, vm: {:.3}, gl: {:.1}",
            self.player_value(1),
            self.player_value(2),
            self.victory_margin(),
            self.game_length(),
        )
    }
}

#[derive(Clone, Copy)]
enum VictoryMarginDirective {
    MinimizeVictoryMargin,
    MaximizeVictoryMargin,
    None,
}

#[cfg(test)]
mod tests;
