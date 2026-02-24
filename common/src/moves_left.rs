pub trait GameLength {
    fn game_length(&self) -> f32;
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

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // map_moves_left_to_one_hot
    // -----------------------------------------------------------------------

    #[test]
    fn map_moves_left_empty_size_returns_empty() {
        assert_eq!(map_moves_left_to_one_hot(10.0, 0), Vec::<f32>::new());
    }

    #[test]
    fn map_moves_left_zero_rounds_to_one() {
        assert_eq!(map_moves_left_to_one_hot(0.0, 4), vec![1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn map_moves_left_one_sets_first_slot() {
        assert_eq!(map_moves_left_to_one_hot(1.0, 4), vec![1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn map_moves_left_two_sets_second_slot() {
        assert_eq!(map_moves_left_to_one_hot(2.0, 4), vec![0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn map_moves_left_rounds_half_up() {
        assert_eq!(map_moves_left_to_one_hot(2.49, 4), vec![0.0, 1.0, 0.0, 0.0]);
        assert_eq!(map_moves_left_to_one_hot(2.50, 4), vec![0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn map_moves_left_clamps_to_size() {
        assert_eq!(
            map_moves_left_to_one_hot(999.0, 4),
            vec![0.0, 0.0, 0.0, 1.0]
        );
    }

    #[test]
    #[should_panic]
    fn map_moves_left_panics_on_nan() {
        let _ = map_moves_left_to_one_hot(f32::NAN, 4);
    }

    #[test]
    #[should_panic]
    fn map_moves_left_panics_on_negative() {
        let _ = map_moves_left_to_one_hot(-1.0, 4);
    }

    #[test]
    #[should_panic]
    fn map_moves_left_panics_on_infinity() {
        let _ = map_moves_left_to_one_hot(f32::INFINITY, 4);
    }

    // -----------------------------------------------------------------------
    // moves_left_expected_value
    // -----------------------------------------------------------------------

    #[test]
    fn moves_left_ev_single_bucket_one_returns_one() {
        let ev = moves_left_expected_value([1.0].into_iter());
        assert!((ev - 1.0).abs() < 1e-6);
    }

    #[test]
    fn moves_left_ev_second_bucket_returns_two() {
        let ev = moves_left_expected_value([0.0, 1.0, 0.0].into_iter());
        assert!((ev - 2.0).abs() < 1e-6);
    }

    #[test]
    fn moves_left_ev_uniform_two_buckets_returns_one_point_five() {
        let ev = moves_left_expected_value([0.5, 0.5].into_iter());
        assert!((ev - 1.5).abs() < 1e-6);
    }

    #[test]
    fn moves_left_ev_empty_returns_zero() {
        let ev = moves_left_expected_value(std::iter::empty());
        assert_eq!(ev, 0.0);
    }
}
