use anyhow::{Result, anyhow};
use common::{
    PropagatedGameLength, PropagatedValue, TranspositionHash, VictoryMarginPropagatedValue,
};
use engine::GameEngine;
use model::GameAnalyzer;
use puct::{EdgeView, PUCT, RollupStats, SelectionPolicy, ValueModel};
use rand::Rng;
use rand::distributions::WeightedIndex;
use rand::prelude::Distribution;
use rand::thread_rng;
use std::time::{Duration, Instant};

use crate::{DirichletOptions, Temperature};
use crate::{EdgeDetails, NodeDetails};
use common::{MovesLeftPropagatedValue, PlayerToMove};
use model::{EdgeMetrics, NodeMetrics};

type SnapshotOf<VM> = <<VM as ValueModel>::Rollup as RollupStats>::Snapshot;

/// Converts a PUCT rollup snapshot into a propagated-values payload.
///
/// The associated type is used so callers don't need to spell out `PV` everywhere.
pub trait SnapshotToPropagated {
    type PropagatedValues: Default;

    fn to_propagated_values(&self, player_to_move: usize) -> Self::PropagatedValues;
}

impl SnapshotToPropagated for puct::MovesLeftSnapshot {
    type PropagatedValues = MovesLeftPropagatedValue;

    fn to_propagated_values(&self, player_to_move: usize) -> Self::PropagatedValues {
        MovesLeftPropagatedValue::new(self.value_for_player(player_to_move), self.game_length())
    }
}

impl SnapshotToPropagated for puct::VictoryMarginSnapshot {
    type PropagatedValues = VictoryMarginPropagatedValue;

    fn to_propagated_values(&self, player_to_move: usize) -> Self::PropagatedValues {
        VictoryMarginPropagatedValue::new(
            self.value_for_player(player_to_move),
            self.victory_margin(),
            0.0,
        )
    }
}

/// Transitional wrapper that runs search via the `puct` crate while providing an
/// MCTS-like container API.
///
/// This is intentionally minimal for now; parity features (focus actions, noise,
/// detailed node/edge introspection) will be layered on next.
pub struct PuctMCTS<'a, E, M, VM, Sel>
where
    E: GameEngine,
    M: GameAnalyzer<State = E::State, Action = E::Action>,
    VM: ValueModel<State = E::State, Predictions = M::Predictions, Terminal = E::Terminal>,
    Sel: SelectionPolicy<SnapshotOf<VM>, State = E::State>,
    E::State: TranspositionHash,
    E::Terminal: engine::Value,
{
    engine: &'a E,
    analyzer: &'a M,
    state: E::State,
    puct: PUCT<'a, E, M, VM, Sel>,
    focus_actions: Vec<E::Action>,
}

impl<'a, E, M, VM, Sel> PuctMCTS<'a, E, M, VM, Sel>
where
    E: GameEngine,
    M: GameAnalyzer<State = E::State, Action = E::Action>,
    VM: ValueModel<State = E::State, Predictions = M::Predictions, Terminal = E::Terminal>,
    Sel: SelectionPolicy<SnapshotOf<VM>, State = E::State>,
    E::State: TranspositionHash,
    E::Terminal: engine::Value,
{
    pub fn new(
        state: E::State,
        engine: &'a E,
        analyzer: &'a M,
        value_model: &'a VM,
        selection: &'a Sel,
    ) -> Self {
        let puct = PUCT::new(engine, analyzer, value_model, selection);

        Self {
            engine,
            analyzer,
            state,
            puct,
            focus_actions: Vec::new(),
        }
    }

    pub fn state(&self) -> &E::State {
        &self.state
    }

    /// PUCT implementation currently does not support persistent root prior mutation.
    /// This is a compatibility stub for legacy callers; it intentionally no-ops.
    pub async fn apply_noise_at_root(&mut self, _dirichlet: Option<&DirichletOptions>) {
        // TODO: Implement true root prior noise once puct exposes prior mutation.
    }

    fn focus_state(&self) -> E::State
    where
        E::State: Clone,
    {
        let mut state = self.state.clone();
        for action in &self.focus_actions {
            state = self.engine.take_action(&state, action);
        }
        state
    }

    pub fn add_focus_to_action(&mut self, action: E::Action) {
        self.focus_actions.push(action);
    }

    pub fn clear_focus(&mut self) {
        self.focus_actions.clear();
    }

    pub fn get_focused_actions(&self) -> &[E::Action] {
        &self.focus_actions
    }

    /// Returns an owned snapshot of focused root edge stats.
    pub fn edge_views(&self) -> Vec<EdgeView<E::Action, SnapshotOf<VM>>>
    where
        E::Action: Clone,
        E::State: Clone,
        SnapshotOf<VM>: Clone,
    {
        let state = self.focus_state();
        self.puct.edge_views(&state)
    }

    /// Runs exactly `simulations` PUCT iterations from the current root.
    pub fn search_simulations(&mut self, simulations: usize)
    where
        E::State: Clone,
    {
        let state = self.focus_state();
        for _ in 0..simulations {
            self.puct.search(&state);
        }
    }

    pub fn num_focus_node_visits(&self) -> usize
    where
        E::Action: Clone,
        E::State: Clone,
        SnapshotOf<VM>: Clone,
    {
        let edges = self.edge_views();
        let sum: u64 = edges.iter().map(|e| e.visits as u64).sum();
        (sum.saturating_add(1)).min(usize::MAX as u64) as usize
    }

    pub fn principal_variation(&self, max_depth: usize) -> Vec<E::Action>
    where
        E::Action: Clone,
        E::State: Clone,
        SnapshotOf<VM>: Clone,
    {
        let mut pv = Vec::new();
        let mut state = self.focus_state();

        for _ in 0..max_depth {
            let edges = self.puct.edge_views(&state);
            let Some(best) = edges
                .into_iter()
                .filter(|e| e.visits > 0)
                .max_by_key(|e| e.visits)
            else {
                break;
            };

            let action = best.action;
            pv.push(action.clone());
            state = self.engine.take_action(&state, &action);
        }

        pv
    }

    pub async fn advance_to_action_retain(&mut self, action: E::Action) -> Result<()> {
        self.state = self.engine.take_action(&self.state, &action);
        self.focus_actions.clear();
        Ok(())
    }

    /// Advances to `action` and prunes the underlying search store to the new root.
    pub async fn advance_to_action(&mut self, action: E::Action) -> Result<()> {
        self.state = self.engine.take_action(&self.state, &action);
        self.focus_actions.clear();
        self.puct.prune(&self.state);
        Ok(())
    }

    pub async fn search_visits(&mut self, visits: usize) -> Result<usize>
    where
        E::Action: Clone,
        E::State: Clone,
        SnapshotOf<VM>: Clone,
    {
        self.search(|node_visits| node_visits < visits).await
    }

    pub async fn search_time_max_visits(
        &mut self,
        duration: Duration,
        max_visits: usize,
    ) -> Result<usize>
    where
        E::Action: Clone,
        E::State: Clone,
        SnapshotOf<VM>: Clone,
    {
        let start = Instant::now();
        self.search(|visits| start.elapsed() < duration && visits < max_visits)
            .await
    }

    pub async fn search<Fn>(&mut self, mut alive: Fn) -> Result<usize>
    where
        Fn: FnMut(usize) -> bool,
        E::Action: Clone,
        E::State: Clone,
        SnapshotOf<VM>: Clone,
    {
        let state = self.focus_state();
        let mut visits = self.num_focus_node_visits();
        let mut depth: usize = 0;

        while alive(visits) {
            self.puct.search(&state);
            visits += 1;
            depth = depth.max(1);
        }

        Ok(depth)
    }

    pub fn select_action<T>(&mut self, temp: &T) -> Result<E::Action>
    where
        T: Temperature<State = E::State>,
        E::Action: Clone,
        E::State: Clone,
        SnapshotOf<VM>: Clone,
    {
        let state = self.focus_state();
        let edges = self.puct.edge_views(&state);

        if edges.is_empty() {
            return Err(anyhow!(
                "Root or focused node does not exist. Run search first."
            ));
        }

        let temp_and_offset = temp.temp(&state);

        if temp_and_offset.temperature > 0.0 {
            let weights = edges.iter().map(|e| {
                (e.visits as f32 + temp_and_offset.temperature_visit_offset)
                    .max(0.0)
                    .powf(1.0 / temp_and_offset.temperature)
            });

            let dist = WeightedIndex::new(weights);
            let chosen_idx = match dist {
                Err(_) => thread_rng().gen_range(0..edges.len()),
                Ok(dist) => dist.sample(&mut thread_rng()),
            };

            return Ok(edges[chosen_idx].action.clone());
        }

        // Temperature == 0: choose the most visited edge.
        edges
            .into_iter()
            .max_by_key(|e| e.visits)
            .map(|e| e.action)
            .ok_or_else(|| anyhow!("No available actions"))
    }

    pub fn get_root_node_metrics(
        &mut self,
    ) -> Result<
        NodeMetrics<
            E::Action,
            M::Predictions,
            <SnapshotOf<VM> as SnapshotToPropagated>::PropagatedValues,
        >,
    >
    where
        E::Action: Clone,
        E::State: Clone + PlayerToMove,
        M::Predictions: Clone,
        SnapshotOf<VM>: Clone + SnapshotToPropagated,
    {
        let state = self.focus_state();
        let analysis = self.analyzer.analyze(&state);
        let predictions = analysis.predictions().clone();

        let edges: Vec<EdgeView<E::Action, SnapshotOf<VM>>> = self.puct.edge_views(&state);
        let player_to_move = state.player_to_move();

        let children = edges
            .into_iter()
            .map(|e| {
                let pv = e
                    .snapshot
                    .as_ref()
                    .map(|s| s.to_propagated_values(player_to_move))
                    .unwrap_or_default();
                EdgeMetrics::new(e.action, e.visits as usize, pv)
            })
            .collect::<Vec<_>>();

        let visits_sum: u64 = children.iter().map(|c| c.visits() as u64).sum();
        let visits = (visits_sum.saturating_add(1)).min(usize::MAX as u64) as usize;

        Ok(NodeMetrics {
            visits,
            predictions,
            children,
        })
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PuctUgiValues {
    value: f32,
    game_length: f32,
}

impl PuctUgiValues {
    pub fn new(value: f32, game_length: f32) -> Self {
        Self { value, game_length }
    }
}

impl PropagatedValue for PuctUgiValues {
    fn value(&self) -> f32 {
        self.value
    }
}

impl PropagatedGameLength for PuctUgiValues {
    fn game_length(&self) -> f32 {
        self.game_length
    }
}

impl Eq for PuctUgiValues {}

impl Ord for PuctUgiValues {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (self.game_length, self.value)
            .partial_cmp(&(other.game_length, other.value))
            .expect("Failed to compare")
    }
}

impl PartialOrd for PuctUgiValues {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

pub trait UgiSnapshot {
    fn value_for_player(&self, player_to_move: usize) -> f32;
    fn game_length(&self) -> f32;
}

impl UgiSnapshot for puct::MovesLeftSnapshot {
    fn value_for_player(&self, player_to_move: usize) -> f32 {
        self.value_for_player(player_to_move)
    }

    fn game_length(&self) -> f32 {
        self.game_length()
    }
}

impl UgiSnapshot for puct::VictoryMarginSnapshot {
    fn value_for_player(&self, player_to_move: usize) -> f32 {
        self.value_for_player(player_to_move)
    }

    fn game_length(&self) -> f32 {
        0.0
    }
}

impl<'a, E, M, VM, Sel> PuctMCTS<'a, E, M, VM, Sel>
where
    E: GameEngine,
    M: GameAnalyzer<State = E::State, Action = E::Action>,
    VM: ValueModel<State = E::State, Predictions = M::Predictions, Terminal = E::Terminal>,
    Sel: SelectionPolicy<SnapshotOf<VM>, State = E::State>,
    E::State: TranspositionHash + Clone,
    E::Terminal: engine::Value,
    E::Action: Clone + PartialEq,
    SnapshotOf<VM>: Clone + UgiSnapshot,
{
    pub fn get_focus_node_details(
        &mut self,
    ) -> Result<Option<NodeDetails<E::Action, PuctUgiValues>>> {
        let state = self.focus_state();
        let edges = self.puct.edge_views(&state);
        if edges.is_empty() {
            return Ok(None);
        }

        let player_to_move = self.engine.player_to_move(&state);
        let edge_details = edges
            .into_iter()
            .map(|e| {
                let propagated_values = e
                    .snapshot
                    .as_ref()
                    .map(|s| {
                        PuctUgiValues::new(s.value_for_player(player_to_move), s.game_length())
                    })
                    .unwrap_or_default();

                let qsa = propagated_values.value();
                EdgeDetails {
                    action: e.action,
                    Nsa: e.visits as usize,
                    Psa: e.policy_prior,
                    Usa: 0.0,
                    cpuct: 0.0,
                    puct_score: qsa,
                    propagated_values,
                }
            })
            .collect::<Vec<_>>();

        let visits_sum: u64 = edge_details.iter().map(|d| d.Nsa as u64).sum();
        let visits = (visits_sum.saturating_add(1)).min(usize::MAX as u64) as usize;

        Ok(Some(NodeDetails {
            visits,
            children: edge_details,
        }))
    }

    pub fn get_principal_variation(
        &mut self,
        action: Option<&E::Action>,
        depth: usize,
    ) -> Result<Vec<EdgeDetails<E::Action, PuctUgiValues>>> {
        let mut state = self.focus_state();
        let player_to_move_root = self.engine.player_to_move(&state);
        let mut pv = Vec::new();

        for ply in 0..depth {
            let edges = self.puct.edge_views(&state);
            if edges.is_empty() {
                break;
            }

            let chosen = if ply == 0 {
                if let Some(desired) = action {
                    edges.into_iter().find(|e| &e.action == desired)
                } else {
                    edges
                        .into_iter()
                        .filter(|e| e.visits > 0)
                        .max_by_key(|e| e.visits)
                }
            } else {
                edges
                    .into_iter()
                    .filter(|e| e.visits > 0)
                    .max_by_key(|e| e.visits)
            };

            let Some(chosen) = chosen else {
                break;
            };

            let propagated_values = chosen
                .snapshot
                .as_ref()
                .map(|s| {
                    PuctUgiValues::new(s.value_for_player(player_to_move_root), s.game_length())
                })
                .unwrap_or_default();
            let qsa = propagated_values.value();

            let details = EdgeDetails {
                action: chosen.action.clone(),
                Nsa: chosen.visits as usize,
                Psa: chosen.policy_prior,
                Usa: 0.0,
                cpuct: 0.0,
                puct_score: qsa,
                propagated_values,
            };

            state = self.engine.take_action(&state, &chosen.action);
            pv.push(details);
        }

        Ok(pv)
    }
}
