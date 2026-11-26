//! Inverse Planning Optimization.

use nalgebra::DVector;

/// Calculates the quadratic objective function cost.
///
/// The cost function penalizes deviations from the prescription dose in the tumor
/// and any dose exceeding the limit in organs at risk (OAR).
///
/// # Arguments
///
/// * `current_dose` - Vector of doses for all voxels (Tumor voxels followed by OAR voxels, or managed by indices).
/// * `tumor_indices` - Indices in `current_dose` corresponding to the tumor.
/// * `organ_indices` - Indices in `current_dose` corresponding to the OAR.
/// * `prescription` ($D_{Rx}$) - Desired dose for the tumor.
/// * `limit` ($D_{lim}$) - Max allowable dose for the organ.
/// * `alpha` - Weighting factor for tumor under/over-dosage.
/// * `beta` - Weighting factor for organ over-dosage.
///
/// # Returns
///
/// * `f64` - The total cost.
#[verified_engine::verified]
pub fn calculate_cost(
    current_dose: &DVector<f64>,
    tumor_indices: &[usize],
    organ_indices: &[usize],
    prescription: f64,
    limit: f64,
    alpha: f64,
    beta: f64,
) -> f64 {
    let mut cost = 0.0;

    // Tumor Term: alpha * sum((D_i - D_Rx)^2)
    for &idx in tumor_indices {
        if idx < current_dose.len() {
            let diff = current_dose[idx] - prescription;
            cost += alpha * diff * diff;
        }
    }

    // Organ Term: beta * sum(H(D_i - D_lim) * (D_i - D_lim)^2)
    for &idx in organ_indices {
        if idx < current_dose.len() {
            let dose = current_dose[idx];
            if dose > limit {
                let diff = dose - limit;
                cost += beta * diff * diff;
            }
        }
    }

    cost
}

/// Performs a single Gradient Descent step to update beamlet weights.
///
/// # Arguments
///
/// * `current_weights` ($w_{old}$) - Current intensity of beamlets.
/// * `gradient` ($\nabla Cost$) - Gradient of the cost function with respect to weights.
/// * `eta` ($\eta$) - Learning rate (step size).
///
/// # Returns
///
/// * `DVector<f64>` - Updated weights ($w_{new}$), clipped to be non-negative.
#[verified_engine::verified]
pub fn update_weights(
    current_weights: &DVector<f64>,
    gradient: &DVector<f64>,
    eta: f64,
) -> DVector<f64> {
    let raw_update = current_weights - (gradient * eta);
    // Constraint: Enforce w >= 0 (clip negative weights)
    raw_update.map(|w| if w < 0.0 { 0.0 } else { w })
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DVector;

    #[test]
    #[verified_engine::verified]
    fn test_cost_function_logic() {
        // Tumor: 2 voxels, indices 0, 1. Prescription 50.
        // Organ: 2 voxels, indices 2, 3. Limit 20.
        // Doses: [50, 40, 10, 30]
        // Tumor cost: (50-50)^2 + (40-50)^2 = 0 + 100 = 100
        // Organ cost: (10 < 20 -> 0) + (30 > 20 -> (30-20)^2 = 100)
        // Total if alpha=1, beta=1 -> 200

        let doses = DVector::from_vec(vec![50.0, 40.0, 10.0, 30.0]);
        let tumor_idx = vec![0, 1];
        let organ_idx = vec![2, 3];

        let cost = calculate_cost(&doses, &tumor_idx, &organ_idx, 50.0, 20.0, 1.0, 1.0);

        assert!((cost - 200.0).abs() < math_commons::registry::TOLERANCE_FAST);

        // Verify organ below limit contributes 0
        // Change organ doses to [10, 15] (both < 20)
        let doses_safe = DVector::from_vec(vec![50.0, 50.0, 10.0, 15.0]);
        let cost_safe = calculate_cost(&doses_safe, &tumor_idx, &organ_idx, 50.0, 20.0, 1.0, 1.0);
        assert_eq!(cost_safe, 0.0);
    }

    #[test]
    #[verified_engine::verified]
    fn test_gradient_update() {
        let w_old = DVector::from_vec(vec![10.0, 5.0]);
        let grad = DVector::from_vec(vec![2.0, -1.0]);
        let eta = 1.0;

        // w_new = [10, 5] - 1.0 * [2, -1] = [8, 6]
        let w_new = update_weights(&w_old, &grad, eta);
        assert!((w_new[0] - 8.0).abs() < math_commons::registry::TOLERANCE_FAST);
        assert!((w_new[1] - 6.0).abs() < math_commons::registry::TOLERANCE_FAST);

        // Test clipping
        let w_old_clip = DVector::from_vec(vec![1.0]);
        let grad_clip = DVector::from_vec(vec![2.0]); // Update -> 1 - 2 = -1
        let w_new_clip = update_weights(&w_old_clip, &grad_clip, 1.0);
        assert_eq!(w_new_clip[0], 0.0);
    }
}
