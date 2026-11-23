use math_commons::error::MathError;
use math_commons::math_kernel::types::Dimension;

/// Calculates the unnormalized joint score $P(I \mid L) P(L)$ for a single voxel label.
///
/// For normalized multi-label posterior probabilities $P(L_i \mid I) = \frac{P(I \mid L_i) P(L_i)}{\sum_j P(I \mid L_j) P(L_j)}$
/// matching the FreeSurfer volumetric segmentation model, use [`normalized_bayesian_posteriors`]
/// or [`bayesian_classify_labels`].
#[verified_engine::verified]
pub fn bayesian_classification(likelihood: f64, prior: f64) -> f64 {
    likelihood * prior
}

/// Computes normalized multi-label Bayesian voxel posteriors in-place with zero heap allocations
/// and returns the Maximum A Posteriori (MAP) tissue label index:
///
/// $$P(L_i \mid I) = \frac{P(I \mid L_i)\,P(L_i)}{\sum_{j=1}^{K} P(I \mid L_j)\,P(L_j)}$$
///
/// # Errors
/// - [`MathError::DimensionMismatch`] if `likelihoods`, `priors`, and `out_posteriors` differ in length or are empty.
/// - [`MathError::NumericalError`] if any likelihood or prior is negative, non-finite, or total marginal evidence is zero.
///
/// # Example
/// ```rust
/// # use domain_applied::__doctest_facade as math_explorer;
/// use math_explorer::applied::freesurfer::normalized_bayesian_posteriors;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let likelihoods = [0.15, 0.80, 0.05]; // CSF, Gray Matter, White Matter
/// let priors = [0.20, 0.50, 0.30];
/// let mut posteriors = [0.0_f64; 3];
/// let map_idx = normalized_bayesian_posteriors(&likelihoods, &priors, &mut posteriors)?;
/// assert_eq!(map_idx, 1);
/// assert!((posteriors.iter().sum::<f64>() - 1.0).abs() < 1e-12);
/// # Ok(())
/// # }
/// ```
#[verified_engine::verified]
pub fn normalized_bayesian_posteriors(
    likelihoods: &[f64],
    priors: &[f64],
    out_posteriors: &mut [f64],
) -> Result<usize, MathError> {
    let k = likelihoods.len();
    if k == 0 || priors.len() != k || out_posteriors.len() != k {
        return Err(MathError::DimensionMismatch {
            expected: Dimension(k.max(1)),
            actual: Dimension(if priors.len() != k {
                priors.len()
            } else {
                out_posteriors.len()
            }),
        });
    }

    let mut evidence = 0.0;
    for i in 0..k {
        let lik = likelihoods[i];
        let pri = priors[i];
        if !lik.is_finite() || !pri.is_finite() || lik < 0.0 || pri < 0.0 {
            return Err(MathError::NumericalError {
                reason: "likelihoods and priors must be finite and non-negative".to_string(),
            });
        }
        let unnorm = bayesian_classification(lik, pri);
        out_posteriors[i] = unnorm;
        evidence += unnorm;
    }

    if !evidence.is_finite() || evidence <= 0.0 {
        return Err(MathError::NumericalError {
            reason: "marginal evidence sum_L P(I|L) P(L) must be strictly positive".to_string(),
        });
    }

    let mut best_idx = 0;
    let mut best_prob = -1.0;
    for (idx, post) in out_posteriors.iter_mut().enumerate() {
        *post /= evidence;
        if *post > best_prob {
            best_prob = *post;
            best_idx = idx;
        }
    }
    Ok(best_idx)
}

/// Allocates and returns `(map_label_index, normalized_posteriors)` across all tissue classes.
#[verified_engine::verified]
pub fn bayesian_classify_labels(
    likelihoods: &[f64],
    priors: &[f64],
) -> Result<(usize, Vec<f64>), MathError> {
    let mut posteriors = vec![0.0_f64; likelihoods.len()];
    let map_idx = normalized_bayesian_posteriors(likelihoods, priors, &mut posteriors)?;
    Ok((map_idx, posteriors))
}

#[cfg(test)]
mod tests {
    #![cfg_attr(any(), verified_engine::verified(opt_out = "tests"))]

    use super::*;
    use verified_engine::engine::VerifiedEngine;

    #[test]
    fn test_normalized_bayesian_posteriors_and_map_selection() {
        let likelihoods = [0.10, 0.70, 0.25];
        let priors = [0.30, 0.40, 0.30];
        let mut posteriors = [0.0_f64; 3];
        let (res, _) = VerifiedEngine::run_verified(|| {
            normalized_bayesian_posteriors(&likelihoods, &priors, &mut posteriors)
        });
        let map_label = res.unwrap();
        assert_eq!(map_label, 1);
        let denom = 0.03 + 0.28 + 0.075;
        assert!((posteriors[0] - 0.03 / denom).abs() < 1e-12);
        assert!((posteriors[1] - 0.28 / denom).abs() < 1e-12);
        assert!((posteriors[2] - 0.075 / denom).abs() < 1e-12);
        assert!((posteriors.iter().sum::<f64>() - 1.0).abs() < 1e-12);

        let (idx2, vec_post) = bayesian_classify_labels(&likelihoods, &priors).unwrap();
        assert_eq!(idx2, 1);
        assert_eq!(vec_post.len(), 3);

        // Zero evidence and negative inputs rejected
        assert!(normalized_bayesian_posteriors(&[0.0, 0.0], &[0.5, 0.5], &mut [0.0; 2]).is_err());
        assert!(normalized_bayesian_posteriors(&[-0.1, 0.5], &[0.5, 0.5], &mut [0.0; 2]).is_err());
    }
}

// [cite:freesurfer]
