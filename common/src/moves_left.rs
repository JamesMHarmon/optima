pub trait GameLength {
    fn game_length_score(&self) -> f32;
}

pub fn moves_left_expected_value<I: Iterator<Item = f32>>(moves_left_scores: I) -> f32 {
    moves_left_scores
        .enumerate()
        .map(|(i, s)| (i + 1) as f32 * s)
        .fold(0.0f32, |sum, element| sum + element)
}

pub fn map_moves_left_to_one_hot(moves_left: f32, moves_left_size: usize) -> Vec<f32> {
    if moves_left_size == 0 {
        return vec![];
    }

    assert!(
        moves_left.is_finite(),
        "Value must be finite (not NaN or infinity)."
    );
    assert!(moves_left >= 0.0, "Value must not be negative.");
    assert!(moves_left <= usize::MAX as f32, "Value must fit in usize.");

    let moves_left = moves_left.round() as usize;
    let moves_left = moves_left.max(1).min(moves_left_size);

    let mut moves_left_one_hot = vec![0f32; moves_left_size];
    moves_left_one_hot[moves_left - 1] = 1.0;

    moves_left_one_hot
}
