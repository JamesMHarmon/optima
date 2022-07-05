#[allow(non_snake_case)]
pub trait QMix<S, V> {
    fn mix_q(game_state: &S, value: &V, q_mix: f32, Q: f32) -> V;
}

#[allow(non_snake_case)]
pub trait ValueStore<S, V>: Default {
    fn get_v_for_player(&self, game_state: &S) -> Option<&V>;

    fn set_v_for_player(&mut self, game_state: &S, V: V);
}
