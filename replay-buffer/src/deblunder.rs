use super::q_mix::{QMix, ValueStore};
use super::sample::PositionMetricsExtended;

#[allow(non_snake_case)]
pub fn deblunder<S, A, V, Vs, Qm>(
    metrics: &mut [PositionMetricsExtended<S, A, V>],
    q_diff_threshold: f32,
    q_diff_width: f32,
) where
    A: PartialEq,
    V: Clone,
    Vs: ValueStore<S, V>,
    Qm: QMix<S, V>,
{
    if q_diff_threshold == 0.0 {
        return;
    }

    let max_move_num = metrics.iter().map(|m| m.move_number).max().unwrap();
    let mut value_stack = ValueStack::<Vs>::new(max_move_num);

    for metric in metrics.iter_mut().rev() {
        let game_state = &metric.metrics.game_state;

        value_stack.set_initial(game_state, &metric.metrics.score);

        let max_visits_child = metric.metrics.policy.child_max_visits();

        value_stack.set_if_not::<S, V, Qm>(game_state, max_visits_child.Q());

        let q_diff = metric.metrics.policy.Q_diff(&metric.chosen_action);
        if q_diff >= q_diff_threshold {
            let q_mix_amt = ((q_diff - q_diff_threshold) / q_diff_width).min(1.0);

            let expected_game_length = max_visits_child.M().round() as usize;

            value_stack.push(q_mix_amt, expected_game_length);

            value_stack.set_if_not::<S, V, Qm>(game_state, max_visits_child.Q());
        }

        let (V, total_moves) = value_stack.latest::<_, V>(game_state);
        metric.metrics.score = V.clone();
        metric.metrics.moves_left =
            (total_moves as isize - metric.move_number as isize).max(1) as usize;
    }
}

struct ValueStack<Vs> {
    v_stores: Vec<(Vs, f32)>,
    total_moves: usize,
}

#[allow(non_snake_case)]
impl<Vs> ValueStack<Vs> {
    fn new<S, V>(max_move_num: usize) -> Self
    where
        Vs: ValueStore<S, V>,
    {
        Self {
            v_stores: vec![(Vs::default(), 0.0)],
            total_moves: max_move_num + 1,
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

        (v, self.total_moves)
    }

    fn push<S, V>(&mut self, q_mix_amt: f32, total_moves: usize)
    where
        Vs: ValueStore<S, V>,
    {
        self.v_stores.push((Vs::default(), q_mix_amt));
        self.total_moves = total_moves;
    }

    fn set_initial<S, V>(&mut self, game_state: &S, V: &V)
    where
        Vs: ValueStore<S, V>,
        V: Clone,
    {
        if let Some((v_store, _)) = self.v_stores.first() {
            if v_store.get_v_for_player(game_state).is_none() {
                self.set_v(game_state, V.clone());
            }
        }
    }

    fn set_if_not<S, V, Qm>(&mut self, game_state: &S, Q: f32)
    where
        Vs: ValueStore<S, V>,
        Qm: QMix<S, V>,
    {
        loop {
            if let Some((_, q_mix_amt)) = self.earliest_unset_v(game_state) {
                let q_mix_amt = *q_mix_amt;
                let latest_value = self
                    .latest_set_v(game_state)
                    .expect("V should be set or provided");
                let mixed_v = Qm::mix_q(game_state, latest_value, q_mix_amt, Q);
                self.set_v(game_state, mixed_v);
            } else {
                return;
            }
        }
    }

    fn earliest_unset_v<S, V>(&mut self, game_state: &S) -> Option<&mut (Vs, f32)>
    where
        Vs: ValueStore<S, V>,
    {
        self.v_stores
            .iter_mut()
            .rev()
            .take_while(|(s, _)| s.get_v_for_player(game_state).is_none())
            .last()
    }

    fn latest_set_v<S, V>(&self, game_state: &S) -> Option<&V>
    where
        Vs: ValueStore<S, V>,
    {
        self.v_stores
            .iter()
            .rev()
            .find_map(|(s, _)| s.get_v_for_player(game_state))
    }

    fn set_v<S, V>(&mut self, game_state: &S, mixed_V: V)
    where
        Vs: ValueStore<S, V>,
    {
        self.earliest_unset_v(game_state)
            .unwrap()
            .0
            .set_v_for_player(game_state, mixed_V);
    }
}
