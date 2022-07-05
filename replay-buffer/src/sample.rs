use engine::GameState;
use half::f16;
use model::PositionMetrics;
use self_play::SelfPlayMetrics;
use tensorflow_model::{Dimension, InputMap, Mode, PolicyMap, ValueMap};

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

        deblunder::<_, _, _, Self::ValueStore, Self>(&mut metrics, q_diff_threshold, q_diff_width);

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

struct PositionMetricsExtended<S, A, V> {
    metrics: PositionMetrics<S, A, V>,
    chosen_action: A,
    move_number: usize,
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

#[allow(non_snake_case)]
fn deblunder<S, A, V, Vs, Qm>(
    metrics: &mut [PositionMetricsExtended<S, A, V>],
    q_diff_threshold: f32,
    q_diff_width: f32,
) where
    A: PartialEq,
    V: Clone,
    Vs: ValueStore<S, V>,
    Qm: QMix<S, V>,
{
    let mut value_stack = ValueStack::<Vs>::new();

    for metric in metrics.iter_mut().rev() {
        let game_state = &metric.metrics.game_state;

        let Q = metric.metrics.policy.child_max_visits().Q();
        value_stack.set_if_not::<S, V, Qm>(game_state, Q);

        let q_diff = metric.metrics.policy.Q_diff(&metric.chosen_action);
        if q_diff >= q_diff_threshold {
            let q_mix_amt = ((q_diff - q_diff_threshold) / q_diff_width).max(1.0);
            value_stack.push(q_mix_amt, metric.metrics.moves_left);

            value_stack.set_if_not::<S, V, Qm>(game_state, Q);
        }

        let (V, moves_left) = value_stack.latest::<_, V>(game_state);
        metric.metrics.score = V.clone();
        metric.metrics.moves_left = moves_left;
    }
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

struct ValueStack<Vs> {
    v_stores: Vec<(Vs, f32)>,
    moves_left: Option<usize>,
}

#[allow(non_snake_case)]
impl<Vs> ValueStack<Vs> {
    fn new<S, V>() -> Self
    where
        Vs: ValueStore<S, V>,
    {
        Self {
            v_stores: vec![(Vs::default(), 0.0)],
            moves_left: None,
        }
    }

    fn latest<S, V>(&self, game_state: &S) -> (&V, usize)
    where
        Vs: ValueStore<S, V>,
    {
        let v = self
            .v_stores
            .last()
            .expect("There should always be at least one V set")
            .0
            .get_v_for_player(game_state)
            .expect("V should always be set before latest is called");

        (v, self.moves_left.expect("Moves left should always be set"))
    }

    fn push<S, V>(&mut self, q_mix_amt: f32, moves_left: usize)
    where
        Vs: ValueStore<S, V>,
    {
        self.v_stores.push((Vs::default(), q_mix_amt));
        self.moves_left = Some(moves_left);
    }

    fn set_if_not<S, V, Qm>(&mut self, game_state: &S, Q: f32)
    where
        Vs: ValueStore<S, V>,
        Qm: QMix<S, V>,
    {
        loop {
            if let Some((_, q_mix_amt)) = self.latest_unset_v(game_state) {
                let q_mix_amt = *q_mix_amt;
                let latest_value = self.latest_set_v(game_state);
                let mixed_v = Qm::mix_q(game_state, latest_value, q_mix_amt, Q);
                self.set_v(game_state, mixed_v);
            } else {
                return;
            }
        }
    }

    fn latest_unset_v<S, V>(&mut self, game_state: &S) -> Option<&mut (Vs, f32)>
    where
        Vs: ValueStore<S, V>,
    {
        self.v_stores
            .iter_mut()
            .rev()
            .take_while(|(s, _)| s.get_v_for_player(game_state).is_none())
            .next()
    }

    fn latest_set_v<S, V>(&self, game_state: &S) -> &V
    where
        Vs: ValueStore<S, V>,
    {
        self.v_stores
            .iter()
            .rev()
            .find_map(|(s, _)| s.get_v_for_player(game_state))
            .expect("There should always be at least one V set")
    }

    fn set_v<S, V>(&mut self, game_state: &S, mixed_V: V)
    where
        Vs: ValueStore<S, V>,
    {
        self.latest_unset_v(game_state)
            .unwrap()
            .0
            .set_v_for_player(game_state, mixed_V);
    }
}
