use engine::GameState;
use half::f16;
use model::{NodeMetrics, PositionMetrics};
use self_play::SelfPlayMetrics;
use tensorflow_model::{Dimension, InputMap, Mode, PolicyMap, ValueMap};

pub trait Sample
where
    Self: InputMap<Self::State>,
    Self: PolicyMap<Self::State, Self::Action, Self::Value>,
    Self: ValueMap<Self::State, Self::Value>,
    Self: Dimension,
{
    type State;
    type Action;
    type Value;

    fn metrics_to_samples(
        &self,
        metrics: SelfPlayMetrics<Self::Action, Self::Value>,
        min_visits: usize,
    ) -> Vec<PositionMetrics<Self::State, Self::Action, Self::Value>>
    where
        Self::State: GameState,
        Self::Value: Clone,
    {
        let mut metrics = get_positions(
            metrics,
            |s| self.move_number(s),
            |s, a| self.take_action(s, a),
        );

        deblunder(&mut metrics);

        filter_full_visits(&mut metrics, min_visits);

        let metrics = metrics.into_iter().filter(|s| self.sample_filter(s));

        metrics
            .flat_map(|position_metrics| self.symmetries(position_metrics))
            .collect()
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

fn get_positions<S, A, V, FM, FA>(
    metrics: SelfPlayMetrics<A, V>,
    move_number: FM,
    take_action: FA,
) -> Vec<PositionMetrics<S, A, V>>
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

        samples.push((
            PositionMetrics {
                game_state: prev_game_state,
                score: score.clone(),
                policy: metrics,
                moves_left: 0,
            },
            move_number,
        ));

        prev_game_state = next_game_state;
    }

    let max_move = samples.iter().map(|(_, m)| *m).max().unwrap();

    for (sample, move_num) in samples.iter_mut() {
        let moves_left = max_move - *move_num + 1;
        sample.moves_left = moves_left;
    }

    let samples = samples.into_iter().map(|(s, _)| s).collect();

    samples
}

fn filter_full_visits<S, A, V>(metrics: &mut Vec<PositionMetrics<S, A, V>>, min_visits: usize) {
    metrics.retain(|m| m.policy.visits >= min_visits)
}

fn deblunder<S, A, V>(metrics: &mut Vec<PositionMetrics<S, A, V>>) {
    // for metric in metrics.iter_mut().rev() {
    //     metric
    // }
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
