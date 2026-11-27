use nalgebra::DMatrix;

/// Module 4.1: Residual Calculation
/// Input: Predicted Noise (final), Added Noise epsilon (from Module 2.3).
/// Operation: Simple subtraction.
/// Output: The raw error tensor in latent space.
#[verified_engine::verified]
pub fn compute_residual(
    predicted_noise: &DMatrix<f64>,
    added_noise: &DMatrix<f64>,
) -> DMatrix<f64> {
    predicted_noise - added_noise
}

/// Module 4.2: Weighting Scheme
/// Input: Raw error grad_2D, Timestep t.
/// Operation: Apply weighting w(t) to normalize gradient magnitude across different noise levels.
/// Output: The weighted gradient vector.
#[verified_engine::verified]
pub fn apply_weighting(residual: &DMatrix<f64>, weight: f64) -> DMatrix<f64> {
    residual * weight
}

/// Module 4.3: The "Detach" Operation (Crucial)
/// Input: delta_SDS.
/// Operation: Stop gradients. We treat this tensor as a constant target for the backward pass.
/// We do not want to backpropagate into the U-Net weights.
/// Output: A fixed gradient tensor ready for backprop.
#[verified_engine::verified]
pub fn detach(tensor: DMatrix<f64>) -> DMatrix<f64> {
    // Identity function in this context, but signifies logical break in gradient graph.
    // In a real framework, this would call .detach().
    tensor
}

/// Computes the full SDS gradient vector (delta_SDS).
#[verified_engine::verified]
pub fn compute_sds_gradient(
    predicted_noise: &DMatrix<f64>,
    added_noise: &DMatrix<f64>,
    weight: f64,
) -> DMatrix<f64> {
    let residual = compute_residual(predicted_noise, added_noise);
    let weighted = apply_weighting(&residual, weight);
    detach(weighted)
}
