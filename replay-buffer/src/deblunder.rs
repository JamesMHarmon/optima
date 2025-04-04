use common::PropagatedValue;
use model::NodeMetrics;

use super::q_mix::{PredictionStore, QMix};
use super::sample::PositionMetricsExtended;

#[allow(non_snake_case)]
pub fn deblunder<S, A, P, PV, Ps, Qm>(
    metrics: &mut [PositionMetricsExtended<S, A, P, PV>],
    q_diff_threshold: f32,
    q_diff_width: f32,
) where
    A: PartialEq,
    P: Clone,
    Ps: PredictionStore<State = S, Predictions = P>,
    Qm: QMix<State = S, Predictions = P, PropagatedValues = PV>,
    PV: PropagatedValue,
{
    if q_diff_threshold == 0.0 {
        return;
    }

    let mut prediction_stack = PredictionStack::<Ps>::new();

    for metric in metrics.iter_mut().rev() {
        let game_state = &metric.metrics.game_state;

        prediction_stack.set_initial(game_state, &metric.target_score);

        let max_visits_child = metric.metrics.policy.child_max_visits();

        prediction_stack
            .set_if_not::<S, P, PV, Qm>(game_state, max_visits_child.propagatedValues());

        let q_diff = q_diff(&metric.metrics.policy, &metric.chosen_action);
        if q_diff >= q_diff_threshold {
            let q_mix_amt = ((q_diff - q_diff_threshold) / q_diff_width).min(1.0);

            prediction_stack.push(q_mix_amt);

            prediction_stack
                .set_if_not::<S, P, PV, Qm>(game_state, max_visits_child.propagatedValues());
        }

        let prediction = prediction_stack.latest::<_, P>(game_state);
        metric.target_score = prediction.clone();
    }
}

struct PredictionStack<Ps> {
    p_stores: Vec<(Ps, f32)>,
}

#[allow(non_snake_case)]
impl<Ps> PredictionStack<Ps> {
    fn new<S, P>() -> Self
    where
        Ps: PredictionStore<State = S, Predictions = P>,
    {
        Self {
            p_stores: vec![(Ps::default(), 0.0)],
        }
    }

    fn latest<S, P>(&self, game_state: &S) -> P
    where
        Ps: PredictionStore<State = S, Predictions = P>,
    {
        let v = self
            .p_stores
            .last()
            .expect("There should always be at least one V set")
            .0
            .get_p_for_player(game_state)
            .expect("V should always be set before latest is called");

        v
    }

    fn push<S, P>(&mut self, q_mix_amt: f32)
    where
        Ps: PredictionStore<State = S, Predictions = P>,
    {
        self.p_stores.push((Ps::default(), q_mix_amt))
    }

    /// Set the initial predictions if they are not already set for the current player.
    fn set_initial<S, P>(&mut self, game_state: &S, predictions: &P)
    where
        Ps: PredictionStore<State = S, Predictions = P>,
        P: Clone,
    {
        if let Some((p_store, _)) = self.p_stores.first() {
            if p_store.get_p_for_player(game_state).is_none() {
                self.set_p(game_state, predictions.clone());
            }
        }
    }

    fn set_if_not<S, P, PV, Qm>(&mut self, game_state: &S, propagated_values: &PV)
    where
        Ps: PredictionStore<State = S, Predictions = P>,
        Qm: QMix<State = S, Predictions = P, PropagatedValues = PV>,
    {
        loop {
            if let Some((_, q_mix_amt)) = self.earliest_unset_p(game_state) {
                let q_mix_amt = *q_mix_amt;
                let latest_p = self
                    .latest_set_p(game_state)
                    .expect("P should be set or provided");
                let mixed_p = Qm::mix_q(game_state, &latest_p, propagated_values, q_mix_amt);
                self.set_p(game_state, mixed_p);
            } else {
                return;
            }
        }
    }

    fn earliest_unset_p<S, P>(&mut self, game_state: &S) -> Option<&mut (Ps, f32)>
    where
        Ps: PredictionStore<State = S, Predictions = P>,
    {
        self.p_stores
            .iter_mut()
            .rev()
            .take_while(|(s, _)| s.get_p_for_player(game_state).is_none())
            .last()
    }

    fn latest_set_p<S, P>(&self, game_state: &S) -> Option<P>
    where
        Ps: PredictionStore<State = S, Predictions = P>,
    {
        self.p_stores
            .iter()
            .rev()
            .find_map(|(s, _)| s.get_p_for_player(game_state))
    }

    fn set_p<S, P>(&mut self, game_state: &S, mixed_P: P)
    where
        Ps: PredictionStore<State = S, Predictions = P>,
    {
        self.earliest_unset_p(game_state)
            .unwrap()
            .0
            .set_p_for_player(game_state, mixed_P);
    }
}

/// Difference between the Q of the specified action and the child that would be played with no temp.
fn q_diff<A, P, PV>(metrics: &NodeMetrics<A, P, PV>, action: &A) -> f32
where
    A: PartialEq,
    PV: PropagatedValue,
{
    let max_visits_q = metrics.child_max_visits().propagatedValues().value();
    let chosen_edge = metrics.children.iter().find(|c| c.action() == action);
    let chosen_q = chosen_edge.expect("Specified action was not found").value();
    max_visits_q - chosen_q
}
