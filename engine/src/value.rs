pub trait Value: Clone {
    fn get_value_for_player(&self, player: usize) -> f32;
}
