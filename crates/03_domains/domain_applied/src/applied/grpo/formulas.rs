const EPSILON_NORM: f64 = 1e-8;

// Re-export moved functions to maintain backward compatibility
pub use super::metrics::pairwise_distance_bleu;
pub use super::rewards::{binary_reward, composite_reward, repetition_penalty, uncertainty_reward};

/// Computes sample mean and sample standard deviation (with Bessel's correction $N - 1$)
/// directly over a borrowed slice without heap allocations.
///
/// Returns `(0.0, 0.0)` for an empty slice and `(rewards[0], 0.0)` for a single-element slice.
#[inline]
fn compute_slice_mean_and_std_dev(rewards: &[f64]) -> (f64, f64) {
    let n = rewards.len();
    if n == 0 {
        return (0.0, 0.0);
    }
    let sum: f64 = rewards.iter().sum();
    let mean = sum / (n as f64);
    if n == 1 {
        return (mean, 0.0);
    }
    let var = rewards
        .iter()
        .map(|&r| {
            let diff = r - mean;
            diff * diff
        })
        .sum::<f64>()
        / ((n - 1) as f64);
    let std_dev = var.max(0.0).sqrt();
    (mean, std_dev)
}

/// Calculates the response-level advantage using z-score normalization.
///
/// # Arguments
///
/// * `rewards` - A slice of scalar rewards for a group of responses.
/// * `reward_i` - The specific reward for which to calculate the advantage.
///
/// # Returns
///
/// The response-level advantage for `reward_i`.
#[verified_engine::verified]
pub fn response_level_advantage(rewards: &[f64], reward_i: f64) -> f64 {
    if rewards.is_empty() {
        return 0.0;
    }
    let (mean, std_dev) = compute_slice_mean_and_std_dev(rewards);
    (reward_i - mean) / (std_dev + EPSILON_NORM)
}

/// Normalizes a full group of rewards in one pass into a pre-allocated output slice without heap allocations.
///
/// This evaluates sample mean and standard deviation once for all responses in $O(G)$ time.
///
/// # Arguments
///
/// * `rewards` - A slice of scalar rewards for a group of responses.
/// * `out` - A mutable destination slice to receive the computed advantages.
#[verified_engine::verified]
pub fn normalize_group_advantages(rewards: &[f64], out: &mut [f64]) {
    if rewards.is_empty() {
        for a in out.iter_mut() {
            *a = 0.0;
        }
        return;
    }
    let (mean, std_dev) = compute_slice_mean_and_std_dev(rewards);
    let denom = std_dev + EPSILON_NORM;
    let n = rewards.len().min(out.len());
    for (r, a) in rewards[..n].iter().zip(out[..n].iter_mut()) {
        *a = (*r - mean) / denom;
    }
    for a in out[n..].iter_mut() {
        *a = 0.0;
    }
}

/// Normalizes a full group of rewards in one pass, returning a vector of advantages.
///
/// This evaluates sample mean and standard deviation once for all responses in $O(G)$ time.
///
/// # Arguments
///
/// * `rewards` - A slice of scalar rewards for a group of responses.
///
/// # Returns
///
/// A `Vec<f64>` containing response-level advantages for the group.
#[verified_engine::verified]
pub fn batch_response_level_advantages(rewards: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; rewards.len()];
    normalize_group_advantages(rewards, &mut out);
    out
}

/// Convenience alias for [`batch_response_level_advantages`].
///
/// # Arguments
///
/// * `rewards` - A slice of scalar rewards for a group of responses.
///
/// # Returns
///
/// A `Vec<f64>` containing response-level advantages for the group.
#[verified_engine::verified]
pub fn batch_group_advantages(rewards: &[f64]) -> Vec<f64> {
    batch_response_level_advantages(rewards)
}

/// The clipped surrogate objective for the policy update in GRPO.
///
/// # Arguments
///
/// * `pi_thetas` - Probabilities of the sampled outputs under the current policy.
/// * `pi_theta_olds` - Probabilities of the sampled outputs under the old policy.
/// * `advantages` - Calculated advantages for the sampled outputs.
/// * `epsilon` - Clipping parameter to limit the policy update step size.
/// * `beta` - Coefficient for the KL divergence term (unused in this formula as written, but part of the general objective).
/// * `kl_divergence` - The KL divergence term.
///
/// # Returns
///
/// The calculated objective value to be maximized (or minimized if negative).
#[verified_engine::verified]
pub fn clipped_surrogate_objective(
    pi_thetas: &[f64],
    pi_theta_olds: &[f64],
    advantages: &[f64],
    epsilon: f64,
    beta: f64,
    kl_divergence: f64,
) -> f64 {
    let g = pi_thetas.len() as f64;
    let sum: f64 = pi_thetas
        .iter()
        .zip(pi_theta_olds.iter())
        .zip(advantages.iter())
        .map(|((pi_theta, pi_theta_old), advantage)| {
            let r_t = pi_theta / pi_theta_old;
            let clipped = r_t.max(1.0 - epsilon).min(1.0 + epsilon);
            (r_t * advantage).min(clipped * advantage)
        })
        .sum();

    -(1.0 / g) * sum + beta * kl_divergence
}

/// The Solver's empirical accuracy for a question x.
///
/// # Arguments
///
/// * `responses` - A slice of response strings.
/// * `pseudo_label` - The label considered correct (the consensus).
///
/// # Returns
///
/// The proportion of responses that match the pseudo-label.
#[verified_engine::verified]
pub fn solver_empirical_accuracy(responses: &[String], pseudo_label: &str) -> f64 {
    let m = responses.len() as f64;
    if m == 0.0 {
        return 0.0;
    }
    let correct_count = responses.iter().filter(|&y| y == pseudo_label).count() as f64;
    correct_count / m
}

/// Lower bound on the KL divergence.
///
/// # Arguments
///
/// * `p` - The probability.
/// * `beta` - A scaling parameter.
///
/// # Returns
///
/// The lower bound on the KL divergence.
#[verified_engine::verified]
pub fn kl_divergence_lower_bound(p: f64, beta: f64) -> f64 {
    if beta == 0.0 {
        return f64::INFINITY;
    }
    p * (1.0 - p) / (2.0 * beta.powi(2))
}
