use engine::GameState;
use half::f16;
use model::PositionMetrics;
use self_play::SelfPlayMetrics;
use tensorflow_model::{Dimension, InputMap, Mode, PolicyMap, ValueMap};

use super::deblunder;
use super::q_mix::{QMix, ValueStore};

pub trait Sample
where
    Self: InputMap<Self::State>,
    Self: PolicyMap<Self::State, Self::Action, Self::Value>,
    Self: ValueMap<Self::State, Self::Value>,
    Self: Dimension,
    Self: QMix<Self::State, Self::Value>,
    Self: Sized,
    Self::ValueStore: ValueStore<Self::State, Self::Value>,
{
    type State;
    type Action;
    type Value;
    type ValueStore;

    fn metrics_to_samples(
        &self,
        metrics: SelfPlayMetrics<Self::Action, Self::Value>,
        min_visits: usize,
        q_diff_threshold: f32,
        q_diff_width: f32,
    ) -> Vec<PositionMetrics<Self::State, Self::Action, Self::Value>>
    where
        Self::State: GameState,
        Self::Value: Clone,
        Self::Action: PartialEq,
    {
        let mut metrics = get_positions(
            metrics,
            |s| self.move_number(s),
            |s, a| self.take_action(s, a),
        );

        deblunder::deblunder::<_, _, _, Self::ValueStore, Self>(
            &mut metrics,
            q_diff_threshold,
            q_diff_width,
        );

        filter_full_visits(&mut metrics, min_visits);

        let metrics = metrics
            .into_iter()
            .filter(|s| self.sample_filter(&s.metrics));

        metrics.flat_map(|s| self.symmetries(s.metrics)).collect()
    }

    fn symmetries(
        &self,
        metric: PositionMetrics<Self::State, Self::Action, Self::Value>,
    ) -> Vec<PositionMetrics<Self::State, Self::Action, Self::Value>> {
        vec![metric]
    }

    fn sample_filter(
        &self,
        _metric: &PositionMetrics<Self::State, Self::Action, Self::Value>,
    ) -> bool {
        true
    }

    fn take_action(&self, game_state: &Self::State, action: &Self::Action) -> Self::State;

    fn move_number(&self, game_state: &Self::State) -> usize;

    fn moves_left_size(&self) -> usize;

    fn input_size(&self) -> usize;

    fn policy_size(&self) -> usize;

    fn metric_to_input_and_targets(
        &self,
        metric: &PositionMetrics<Self::State, Self::Action, Self::Value>,
    ) -> InputAndTargets {
        let policy_output =
            self.policy_metrics_to_expected_output(&metric.game_state, &metric.policy);
        let value_output = self.map_value_to_value_output(&metric.game_state, &metric.score);
        let moves_left_output =
            map_moves_left_to_one_hot(metric.moves_left, self.moves_left_size());
        let input_len = self.input_size();

        let sum_of_policy = policy_output.iter().filter(|&&x| x >= 0.0).sum::<f32>();
        assert!(
            f32::abs(sum_of_policy - 1.0) <= f32::EPSILON * policy_output.len() as f32,
            "Policy output should sum to 1.0 but actual sum is {}",
            sum_of_policy
        );

        let mut input = vec![f16::ZERO; input_len];
        self.game_state_to_input(&metric.game_state, &mut input, Mode::Train);
        let input = input.into_iter().map(f16::to_f32).collect();

        assert!(
            (-1.0..=1.0).contains(&value_output),
            "Value output should be in range -1.0-1.0 but was {}",
            &value_output
        );

        InputAndTargets {
            input,
            policy_output,
            value_output,
            moves_left_output,
        }
    }
}

pub struct InputAndTargets {
    pub input: Vec<f32>,
    pub policy_output: Vec<f32>,
    pub value_output: f32,
    pub moves_left_output: Vec<f32>,
}

pub struct PositionMetricsExtended<S, A, V> {
    pub metrics: PositionMetrics<S, A, V>,
    pub chosen_action: A,
    pub move_number: usize,
}

fn get_positions<S, A, V, FM, FA>(
    metrics: SelfPlayMetrics<A, V>,
    move_number: FM,
    take_action: FA,
) -> Vec<PositionMetricsExtended<S, A, V>>
where
    S: GameState,
    V: Clone,
    FM: Fn(&S) -> usize,
    FA: Fn(&S, &A) -> S,
{
    let (analysis, score) = metrics.into_inner();

    let mut prev_game_state = S::initial();
    let mut samples = vec![];
    for (action, metrics) in analysis {
        let next_game_state = take_action(&prev_game_state, &action);
        let move_number = move_number(&prev_game_state);

        samples.push(PositionMetricsExtended {
            metrics: PositionMetrics {
                game_state: prev_game_state,
                score: score.clone(),
                policy: metrics,
                moves_left: 0,
            },
            chosen_action: action,
            move_number,
        });

        prev_game_state = next_game_state;
    }

    let max_move = samples.iter().map(|m| m.move_number).max().unwrap();

    for metrics in samples.iter_mut() {
        let moves_left = max_move - metrics.move_number + 1;
        metrics.metrics.moves_left = moves_left;
    }

    samples
}

fn filter_full_visits<S, A, V>(
    metrics: &mut Vec<PositionMetricsExtended<S, A, V>>,
    min_visits: usize,
) {
    metrics.retain(|m| m.metrics.policy.visits >= min_visits)
}

fn map_moves_left_to_one_hot(moves_left: usize, moves_left_size: usize) -> Vec<f32> {
    if moves_left_size == 0 {
        return vec![];
    }

    let moves_left = moves_left.max(0).min(moves_left_size);
    let mut moves_left_one_hot = vec![0f32; moves_left_size];
    moves_left_one_hot[moves_left - 1] = 1.0;

    moves_left_one_hot
}
