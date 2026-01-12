use std::collections::HashMap;

use common::PropagatedValue;
use engine::GameState;
use half::f16;
use model::PositionMetrics;
use self_play::SelfPlayMetrics;
use tensorflow_model::{InputMap, Mode, PredictionsMap};

use super::deblunder;
use super::q_mix::{PredictionStore, QMix};

#[allow(clippy::type_complexity)]
pub trait Sample {
    type State;
    type Action;
    type Predictions;
    type PropagatedValues;
    type PredictionStore;

    fn metrics_to_samples(
        &self,
        metrics: SelfPlayMetrics<
            <Self as Sample>::Action,
            <Self as Sample>::Predictions,
            <Self as Sample>::PropagatedValues,
        >,
        min_visits: usize,
        q_diff_threshold: f32,
        q_diff_width: f32,
    ) -> Vec<
        PositionMetricsExtended<
            <Self as Sample>::State,
            <Self as Sample>::Action,
            <Self as Sample>::Predictions,
            <Self as Sample>::PropagatedValues,
        >,
    >
    where
        Self: Sized,
        <Self as Sample>::State: GameState,
        <Self as Sample>::Action: PartialEq + Clone,
        <Self as Sample>::Predictions: Clone,
        <Self as Sample>::PropagatedValues: PropagatedValue,
        Self::PredictionStore: PredictionStore<
                State = <Self as Sample>::State,
                Predictions = <Self as Sample>::Predictions,
            >,
        Self: QMix<
                State = <Self as Sample>::State,
                Predictions = <Self as Sample>::Predictions,
                PropagatedValues = <Self as Sample>::PropagatedValues,
            >,
    {
        let mut metrics = get_positions(metrics, |s, a| self.take_action(s, a));

        deblunder::deblunder::<_, _, _, _, <Self as Sample>::PredictionStore, Self>(
            &mut metrics,
            q_diff_threshold,
            q_diff_width,
        );

        filter_full_visits(&mut metrics, min_visits);

        let metrics = metrics
            .into_iter()
            .filter(|s| self.sample_filter(&s.metrics));

        metrics
            .flat_map(|s| {
                self.symmetries(s.metrics)
                    .into_iter()
                    .map(move |metrics| PositionMetricsExtended {
                        metrics,
                        target_score: s.target_score.clone(),
                        chosen_action: s.chosen_action.clone(),
                    })
            })
            .collect()
    }

    fn symmetries(
        &self,
        metric: PositionMetrics<
            <Self as Sample>::State,
            <Self as Sample>::Action,
            <Self as Sample>::Predictions,
            <Self as Sample>::PropagatedValues,
        >,
    ) -> Vec<
        PositionMetrics<
            <Self as Sample>::State,
            <Self as Sample>::Action,
            <Self as Sample>::Predictions,
            <Self as Sample>::PropagatedValues,
        >,
    >;

    fn sample_filter(
        &self,
        _metric: &PositionMetrics<
            <Self as Sample>::State,
            <Self as Sample>::Action,
            <Self as Sample>::Predictions,
            <Self as Sample>::PropagatedValues,
        >,
    ) -> bool {
        true
    }

    fn take_action(
        &self,
        game_state: &<Self as Sample>::State,
        action: &<Self as Sample>::Action,
    ) -> <Self as Sample>::State;

    fn input_size(&self) -> usize;

    fn input_values<'a>(&self, input_and_targets: &'a InputAndTargets) -> &'a [f32] {
        &input_and_targets.values[..self.input_size()]
    }

    fn outputs(&self) -> Vec<(String, usize)>;

    fn output_values<'a>(&self, input_and_targets: &'a InputAndTargets, name: &str) -> &'a [f32] {
        let (offset, size) = self.output_offset_and_size(name);
        &input_and_targets.values[offset..offset + size]
    }

    fn output_offset_and_size(&self, name: &str) -> (usize, usize) {
        let mut offset = self.input_size();
        for (n, size) in self.outputs() {
            if n == name {
                return (offset, size);
            }
            offset += size;
        }
        panic!("Output not found");
    }

    fn sample_size(&self) -> usize {
        self.input_size() + self.outputs().iter().map(|(_, size)| size).sum::<usize>()
    }

    fn metric_to_input_and_targets(
        &self,
        targets: <Self as Sample>::Predictions,
        metric: &PositionMetrics<
            <Self as Sample>::State,
            <Self as Sample>::Action,
            <Self as Sample>::Predictions,
            <Self as Sample>::PropagatedValues,
        >,
    ) -> InputAndTargets
    where
        Self: InputMap<State = <Self as Sample>::State>,
        Self: PredictionsMap<
                State = <Self as Sample>::State,
                Action = <Self as Sample>::Action,
                Predictions = <Self as Sample>::Predictions,
                PropagatedValues = <Self as Sample>::PropagatedValues,
            >,
    {
        let targets = self.to_output(&metric.game_state, targets, &metric.node_metrics);
        let input_len = self.input_size();

        let mut input = vec![f16::ZERO; input_len];
        self.game_state_to_input(&metric.game_state, &mut input, Mode::Train);
        let input = input.into_iter().map(f16::to_f32).collect::<Vec<f32>>();

        self.input_and_targets(&input, &targets)
    }

    fn input_and_targets(
        &self,
        input: &[f32],
        targets: &HashMap<String, Vec<f32>>,
    ) -> InputAndTargets {
        let mut values = Vec::with_capacity(self.sample_size());

        assert_eq!(input.len(), self.input_size(), "Input size mismatch");
        values.extend_from_slice(input);

        for (name, size) in self.outputs() {
            let output_vals = targets.get(&name).expect("Target not found");
            assert_eq!(output_vals.len(), size, "Target size mismatch");
            values.extend_from_slice(output_vals);
        }

        InputAndTargets { values }
    }
}

pub struct InputAndTargets {
    values: Vec<f32>,
}

impl InputAndTargets {
    pub fn as_slice(&self) -> &[f32] {
        &self.values
    }
}

impl From<Vec<f32>> for InputAndTargets {
    fn from(values: Vec<f32>) -> Self {
        InputAndTargets { values }
    }
}

pub struct PositionMetricsExtended<S, A, P, PV> {
    pub metrics: PositionMetrics<S, A, P, PV>,
    pub target_score: P,
    pub chosen_action: A,
}

fn get_positions<S, A, P, PV, FA>(
    metrics: SelfPlayMetrics<A, P, PV>,
    take_action: FA,
) -> Vec<PositionMetricsExtended<S, A, P, PV>>
where
    S: GameState,
    P: Clone,
    FA: Fn(&S, &A) -> S,
{
    let (analysis, score) = metrics.into_inner();

    let mut pre_action_game_state = S::initial();
    let mut samples = vec![];
    for (action, metrics) in analysis {
        let post_action_game_state = take_action(&pre_action_game_state, &action);

        samples.push(PositionMetricsExtended {
            metrics: PositionMetrics {
                game_state: pre_action_game_state,
                node_metrics: metrics,
            },
            chosen_action: action,
            target_score: score.clone(),
        });

        pre_action_game_state = post_action_game_state;
    }

    samples
}

fn filter_full_visits<S, A, P, PV>(
    metrics: &mut Vec<PositionMetricsExtended<S, A, P, PV>>,
    min_visits: usize,
) {
    metrics.retain(|m| m.metrics.node_metrics.visits >= min_visits)
}
