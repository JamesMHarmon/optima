#[allow(non_snake_case)]
pub trait QMix {
    type State;
    type Predictions;
    type PropagatedValues;

    fn mix_q(
        game_state: &Self::State,
        post_blunder_prediction: &Self::Predictions,
        pre_blunder_propagated_values: &Self::PropagatedValues,
        q_mix: f32,
    ) -> Self::Predictions;
}

#[allow(non_snake_case)]
pub trait PredictionStore: Default {
    type State;
    type Predictions;

    fn get_p_for_player(&self, game_state: &Self::State) -> Option<&Self::Predictions>;

    fn set_p_for_player(&mut self, game_state: &Self::State, prediction: Self::Predictions);
}
