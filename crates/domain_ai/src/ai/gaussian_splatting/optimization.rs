#[derive(Debug, PartialEq, Clone, Copy, Eq)]
#[allow(missing_docs)]
pub enum DensityAction {
    #[allow(missing_docs)]
    Clone,
    #[allow(missing_docs)]
    Split,
    #[allow(missing_docs)]
    Prune,
    #[allow(missing_docs)]
    Keep,
}

/// Determines whether a Gaussian should be cloned, split, or pruned.
///
/// Based on gradient magnitude, scale, and opacity thresholds.
///
/// * `grad_magnitude`: The magnitude of the positional gradient for the Gaussian.
/// * `scale_max`: The maximum scale component of the Gaussian.
/// * `opacity`: The opacity of the Gaussian.
/// * `grad_threshold`: Threshold for positional gradient (tau_pos).
/// * `scale_threshold`: Threshold for scale to decide between split vs clone (phi).
/// * `opacity_threshold`: Minimum opacity threshold for pruning.
#[verified_engine::verified]
pub fn determine_density_action(
    grad_magnitude: f64,
    scale_max: f64,
    opacity: f64,
    grad_threshold: f64,
    scale_threshold: f64,
    opacity_threshold: f64,
) -> DensityAction {
    // Prune if opacity is too low
    if opacity < opacity_threshold {
        return DensityAction::Prune;
    }

    // Check for densification
    if grad_magnitude > grad_threshold {
        if scale_max > scale_threshold {
            return DensityAction::Split;
        } else {
            return DensityAction::Clone;
        }
    }

    DensityAction::Keep
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[verified_engine::verified]
    fn test_density_actions() {
        // Prune
        assert_eq!(
            determine_density_action(0.0, 1.0, 0.001, 0.1, 1.0, 0.01),
            DensityAction::Prune
        );

        // Clone (High grad, small scale)
        assert_eq!(
            determine_density_action(0.2, 0.5, 0.5, 0.1, 1.0, 0.01),
            DensityAction::Clone
        );

        // Split (High grad, large scale)
        assert_eq!(
            determine_density_action(0.2, 1.5, 0.5, 0.1, 1.0, 0.01),
            DensityAction::Split
        );

        // Keep
        assert_eq!(
            determine_density_action(0.05, 0.5, 0.5, 0.1, 1.0, 0.01),
            DensityAction::Keep
        );
    }
}
