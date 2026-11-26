/// The uncertainty reward function.
///
/// # Arguments
///
/// * `p_hat` - The estimated probability of the model's prediction.
///
/// # Returns
///
/// The uncertainty reward, which is higher when `p_hat` is close to 0.5.
#[verified_engine::verified]
pub fn uncertainty_reward(p_hat: f64) -> f64 {
    1.0 - 2.0 * (p_hat - 0.5).abs()
}

/// The repetition penalty for a question.
///
/// # Arguments
///
/// * `cluster_size` - The size of the cluster the question belongs to.
/// * `batch_size` - The total batch size.
/// * `lambda` - A scaling factor for the penalty.
///
/// # Returns
///
/// The calculated repetition penalty.
#[verified_engine::verified]
pub fn repetition_penalty(cluster_size: usize, batch_size: usize, lambda: f64) -> f64 {
    lambda * (cluster_size as f64 / batch_size as f64)
}

/// The composite reward for a valid question.
///
/// # Arguments
///
/// * `uncertainty_reward` - The calculated uncertainty reward.
/// * `repetition_penalty` - The calculated repetition penalty.
///
/// # Returns
///
/// The composite reward, which is non-negative.
#[verified_engine::verified]
pub fn composite_reward(uncertainty_reward: f64, repetition_penalty: f64) -> f64 {
    (uncertainty_reward - repetition_penalty).max(0.0)
}

/// The binary reward for a generation x_i.
///
/// # Arguments
///
/// * `satisfies_check` - A boolean indicating if the generation satisfies the correctness check.
///
/// # Returns
///
/// 1 if the check is satisfied, 0 otherwise.
#[verified_engine::verified]
pub fn binary_reward(satisfies_check: bool) -> i32 {
    if satisfies_check { 1 } else { 0 }
}
