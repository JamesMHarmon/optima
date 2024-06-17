use model::NodeMetrics;

use super::q_mix::{QMix, PredictionStore};
use super::sample::PositionMetricsExtended;

#[allow(non_snake_case)]
pub fn deblunder<S, A, P, PV, Ps, Qm>(
    metrics: &mut [PositionMetricsExtended<S, A, P, PV>],
    q_diff_threshold: f32,
    q_diff_width: f32,
) where
    A: PartialEq,
    P: Clone,
    Ps: PredictionStore<S, P>,
    Qm: QMix<S, P>,
{
    if q_diff_threshold == 0.0 {
        return;
    }

    let mut prediction_stack = PredictionStack::<Ps>::new();

    for metric in metrics.iter_mut().rev() {
        let game_state = &metric.metrics.game_state;

        prediction_stack.set_initial(game_state, &metric.target_score);

        let max_visits_child = metric.metrics.policy.child_max_visits();

        prediction_stack.set_if_not::<S, P, Qm>(game_state, max_visits_child.Q());

        let q_diff = q_diff(metric.metrics.policy, &metric.chosen_action);
        if q_diff >= q_diff_threshold {
            let q_mix_amt = ((q_diff - q_diff_threshold) / q_diff_width).min(1.0);

            prediction_stack.push(q_mix_amt);

            prediction_stack.set_if_not::<S, P, Qm>(game_state, max_visits_child.Q());
        }

        let (prediction, total_moves) = prediction_stack.latest::<_, P>(game_state);
        metric.metrics.score = prediction.clone();
        metric.metrics.moves_left =
            (total_moves as isize - metric.move_number as isize).max(1) as usize;
    }
}

struct PredictionStack<Ps> {
    p_stores: Vec<(Ps, f32)>
}

#[allow(non_snake_case)]
impl<Ps> PredictionStack<Ps> {
    fn new<S, P>() -> Self
    where
        Ps: PredictionStore<S, P>,
    {
        Self {
            p_stores: vec![(Ps::default(), 0.0)]
        }
    }

    fn latest<S, P>(&self, game_state: &S) -> (&P, usize)
    where
        Ps: PredictionStore<S, P>,
    {
        let v = self
            .p_stores
            .last()
            .expect("There should always be at least one V set")
            .0
            .get_p_for_player(game_state)
            .expect("V should always be set before latest is called");

        (v, self.total_moves)
    }

    fn push<S, P>(&mut self, q_mix_amt: f32)
    where
        Ps: PredictionStore<S, P>,
    {
        self.p_stores.push((Ps::default(), q_mix_amt))
    }

    fn set_initial<S, P>(&mut self, game_state: &S, predictions: &P)
    where
        Ps: PredictionStore<S, P>,
        P: Clone,
    {
        if let Some((p_store, _)) = self.p_stores.first() {
            if p_store.get_p_for_player(game_state).is_none() {
                self.set_p(game_state, predictions.clone());
            }
        }
    }

    fn set_if_not<S, P, Qm>(&mut self, game_state: &S, predictions: &P)
    where
        Ps: PredictionStore<S, P>,
        Qm: QMix<S, P>,
    {
        loop {
            if let Some((_, q_mix_amt)) = self.earliest_unset_p(game_state) {
                let q_mix_amt = *q_mix_amt;
                let latest_p = self
                    .latest_set_p(game_state)
                    .expect("P should be set or provided");
                let mixed_p = Qm::mix_q(game_state, latest_p, &predictions, q_mix_amt);
                self.set_p(game_state, mixed_p);
            } else {
                return;
            }
        }
    }

    fn earliest_unset_p<S, P>(&mut self, game_state: &S) -> Option<&mut (Ps, f32)>
    where
        Ps: PredictionStore<S, P>,
    {
        self.p_stores
            .iter_mut()
            .rev()
            .take_while(|(s, _)| s.get_p_for_player(game_state).is_none())
            .last()
    }

    fn latest_set_p<S, P>(&self, game_state: &S) -> Option<&P>
    where
        Ps: PredictionStore<S, P>,
    {
        self.p_stores
            .iter()
            .rev()
            .find_map(|(s, _)| s.get_p_for_player(game_state))
    }

    fn set_p<S, P>(&mut self, game_state: &S, mixed_P: P)
    where
        Ps: PredictionStore<S, P>,
    {
        self.earliest_unset_p(game_state)
            .unwrap()
            .0
            .set_p_for_player(game_state, mixed_P);
    }
}

// @TODO: Was a method on the NodeMetrics struct
/// Difference between the Q of the specified action and the child that would be played with no temp.
fn q_diff<A, P, PV>(&metrics: NodeMetrics<A, P, PV>, action: &A) -> f32
where
    A: PartialEq,
{
    let max_visits_Q = metrics.child_max_visits().Q();
    let chosen_Q = metrics.children.iter().find(|c| c.action() == action);
    let chosen_Q = chosen_Q.expect("Specified action was not found").Q();
    max_visits_Q - chosen_Q
}
