use super::analytics::ActionWithPolicy;
use common::softmax::softmax;
use half::f16;

pub fn update_logit_policies_to_softmax<A>(logit_policies: &mut [ActionWithPolicy<A>]) {
    let temperature = 1.2;
    let policy_scores_by_action = logit_policies
        .iter()
        .map(|p| p.policy_score.into())
        .collect::<Vec<_>>();
    let softmaxed_policies = softmax(&policy_scores_by_action, temperature);

    for (awp, policy) in logit_policies.iter_mut().zip(softmaxed_policies) {
        awp.policy_score = f16::from_f32(policy);
    }
}
