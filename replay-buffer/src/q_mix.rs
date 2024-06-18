#[allow(non_snake_case)]
pub trait QMix<S, P, PV> {
    fn mix_q(
        game_state: &S,
        post_blunder_prediction: &P,
        pre_blunder_propagated_values: &PV,
        q_mix: f32,
    ) -> P;
}

#[allow(non_snake_case)]
pub trait PredictionStore<S, P>: Default {
    fn get_v_for_player(&self, game_state: &S) -> Option<&P>;

    fn set_v_for_player(&mut self, game_state: &S, prediction: P);
}
