#[allow(non_snake_case)]
pub trait QMix<S, P> {
    fn mix_q(game_state: &S, latest_prediction: &P, other_prediction: &P, q_mix: f32) -> P;
}

#[allow(non_snake_case)]
pub trait PredictionStore<S, P>: Default {
    fn get_v_for_player(&self, game_state: &S) -> Option<&P>;

    fn set_v_for_player(&mut self, game_state: &S, prediction: P);
}
