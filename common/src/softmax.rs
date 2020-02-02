// (exp(p-max_p))^(1/T) = exp((p-max_p)/T).
pub fn softmax(logits: &[f32], temperature: f32) -> Vec<f32> {
    let max_p = logits.iter().cloned().fold(std::f32::MIN, f32::max);
    let softmaxed = logits.iter().map(|&p| ((p - max_p) / temperature).exp()).collect::<Vec<_>>();
    let sum = softmaxed.iter().sum::<f32>();
    let reduced = softmaxed.iter().map(|p| p / sum).collect::<Vec<_>>();

    reduced
}

#[cfg(test)]
mod test {
    use super::softmax;
    use assert_approx_eq::assert_approx_eq;

    #[test]
    fn test_softmax_temp_1() {
        let logits = vec![0.1, 0.2, 0.3, 0.1];
        let temperature = 1.0;
        let expected = vec![0.231129, 0.255437, 0.282302, 0.231129];
        let actual = softmax(&logits, temperature);

        for (l, r) in expected.iter().zip(actual) {
            assert_approx_eq!(l, r, 0.00001);
        }
    }

    #[test]
    fn test_softmax_temp_1_1() {
        let logits = vec![0.1, 0.2, 0.3, 0.1];
        let temperature = 1.1;
        let expected = vec![0.232852, 0.255012, 0.279282, 0.232852];
        let actual = softmax(&logits, temperature);

        for (l, r) in expected.iter().zip(actual) {
            assert_approx_eq!(l, r, 0.00001);
        }
    }

    #[test]
    fn test_softmax_temp_1_2() {
        let logits = vec![0.1, 0.2, 0.3, 0.1];
        let temperature = 1.2;
        let expected = vec![0.234287, 0.254647, 0.276777, 0.234287];
        let actual = softmax(&logits, temperature);

        for (l, r) in expected.iter().zip(actual) {
            assert_approx_eq!(l, r, 0.00001);
        }
    }

    #[test]
    fn test_softmax_temp_all_0() {
        let logits = vec![0.0, 0.0];
        let temperature = 1.2;
        let expected = vec![0.5, 0.5];
        let actual = softmax(&logits, temperature);

        for (l, r) in expected.iter().zip(actual) {
            assert_approx_eq!(l, r, 0.00001);
        }
    }

    #[test]
    fn test_softmax_temp_one_0() {
        let logits = vec![0.1, 1.5, 0.2, 0.0];
        let temperature = 1.2;
        let expected = vec![0.160817, 0.516429, 0.174793, 0.147959];
        let actual = softmax(&logits, temperature);

        for (l, r) in expected.iter().zip(actual) {
            assert_approx_eq!(l, r, 0.00001);
        }
    }

    #[test]
    fn test_softmax_temp_single_0() {
        let logits = vec![0.0];
        let temperature = 1.2;
        let expected = vec![1.0];
        let actual = softmax(&logits, temperature);

        for (l, r) in expected.iter().zip(actual) {
            assert_approx_eq!(l, r, 0.00001);
        }
    }

    #[test]
    fn test_softmax_temp_empty() {
        let logits = vec![];
        let temperature = 1.2;
        let expected = vec![0.234287, 0.254647, 0.276777, 0.234287];
        let actual = softmax(&logits, temperature);

        for (l, r) in expected.iter().zip(actual) {
            assert_approx_eq!(l, r, 0.00001);
        }
    }

    #[test]
    fn test_softmax_temp_singular() {
        let logits = vec![0.3];
        let temperature = 1.2;
        let expected = vec![1.0];
        let actual = softmax(&logits, temperature);

        for (l, r) in expected.iter().zip(actual) {
            assert_approx_eq!(l, r, 0.00001);
        }
    }
}
