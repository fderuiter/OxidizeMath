//! Calibrated summary statistics, normal quantiles, and Kong effective sample size helpers.

use super::MonteCarloError;
use statrs::function::erf::{erfc, erfc_inv};

/// Evaluates the standard normal cumulative distribution function $\Phi(z)$ without heap allocation.
#[inline]
#[verified_engine::verified]
pub fn normal_cdf_fast(z: f64) -> f64 {
    0.5 * erfc(-z * std::f64::consts::FRAC_1_SQRT_2)
}

/// Evaluates the standard normal quantile $\Phi^{-1}(p)$ for $p \in (0, 1)$ without heap allocation.
#[inline]
#[verified_engine::verified]
pub fn normal_inv_cdf_fast(p: f64) -> f64 {
    let clamped = p.clamp(1e-15, 1.0 - 1e-15);
    -std::f64::consts::SQRT_2 * erfc_inv(2.0 * clamped)
}

/// Validates that `confidence_level` lies strictly inside `(0.0, 1.0)` and returns $z_{1 - \alpha/2}$.
#[inline]
#[verified_engine::verified]
pub fn two_sided_normal_z(confidence_level: f64) -> Result<f64, MonteCarloError> {
    if !confidence_level.is_finite() || confidence_level <= 0.0 || confidence_level >= 1.0 {
        return Err(MonteCarloError::InvalidConfidenceLevel {
            confidence: confidence_level,
        });
    }
    let upper_prob = 0.5 + 0.5 * confidence_level;
    Ok(normal_inv_cdf_fast(upper_prob))
}

/// Computes Kong's (1992) Effective Sample Size $\text{ESS} = \frac{(\sum w_i)^2}{\sum w_i^2}$.
///
/// Flags non-finite, negative, all-zero, or collapsed weights via [`MonteCarloError`].
#[verified_engine::verified]
pub fn effective_sample_size(weights: &[f64]) -> Result<f64, MonteCarloError> {
    if weights.is_empty() {
        return Err(MonteCarloError::InvalidSampleCount { samples: 0 });
    }
    let mut sum_w = 0.0;
    let mut sum_w_sq = 0.0;
    for &w in weights {
        if !w.is_finite() || w < 0.0 {
            return Err(MonteCarloError::NonFiniteEvaluation);
        }
        sum_w += w;
        sum_w_sq += w * w;
    }
    if sum_w <= 0.0 || sum_w_sq <= 0.0 || !sum_w.is_finite() || !sum_w_sq.is_finite() {
        return Err(MonteCarloError::DegenerateWeights {
            effective_sample_size: 0.0,
        });
    }
    let ess = (sum_w * sum_w) / sum_w_sq;
    if !ess.is_finite() || (weights.len() >= 2 && ess < 1.5) {
        return Err(MonteCarloError::DegenerateWeights {
            effective_sample_size: ess,
        });
    }
    Ok(ess)
}

/// Calibrated Monte Carlo estimator summary with standard error, confidence bounds, and `ESS`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MonteCarloEstimate {
    /// Point estimate of the expectation or integral.
    pub mean: f64,
    /// Unbiased sample variance of the underlying estimator draws.
    pub variance: f64,
    /// Standard error of `mean`.
    pub standard_error: f64,
    /// Lower confidence bound at `confidence_level`.
    pub ci_lower: f64,
    /// Upper confidence bound at `confidence_level`.
    pub ci_upper: f64,
    /// Nominal two-sided confidence level in `(0.0, 1.0)`.
    pub confidence_level: f64,
    /// Total function evaluations consumed.
    pub samples: usize,
    /// Effective sample size (`ESS`). Equals `samples as f64` for unweighted IID draws.
    pub effective_sample_size: f64,
}

impl MonteCarloEstimate {
    /// Builds a calibrated [`MonteCarloEstimate`] from raw estimator values using Welford's pass.
    #[verified_engine::verified]
    pub fn from_samples(values: &[f64], confidence_level: f64) -> Result<Self, MonteCarloError> {
        if values.is_empty() {
            return Err(MonteCarloError::InvalidSampleCount { samples: 0 });
        }
        let z = two_sided_normal_z(confidence_level)?;
        let (mean, m2) = welford_mean_m2(values)?;
        let n = values.len();
        let variance = if n > 1 { m2 / ((n - 1) as f64) } else { 0.0 };
        let standard_error = if n > 1 {
            (variance / (n as f64)).sqrt()
        } else {
            0.0
        };
        let half = z * standard_error;
        Ok(Self {
            mean,
            variance,
            standard_error,
            ci_lower: mean - half,
            ci_upper: mean + half,
            confidence_level,
            samples: n,
            effective_sample_size: n as f64,
        })
    }

    /// Half-width `(ci_upper - ci_lower) * 0.5` of the two-sided confidence interval.
    #[inline]
    #[verified_engine::verified]
    pub fn ci_half_width(&self) -> f64 {
        0.5 * (self.ci_upper - self.ci_lower)
    }
}

/// Numerically stable Welford accumulation returning `(mean, sum_squared_deviations)`.
#[verified_engine::verified]
pub(crate) fn welford_mean_m2(values: &[f64]) -> Result<(f64, f64), MonteCarloError> {
    let mut mean = 0.0;
    let mut m2 = 0.0;
    for (idx, &x) in values.iter().enumerate() {
        if !x.is_finite() {
            return Err(MonteCarloError::NonFiniteEvaluation);
        }
        let k = (idx + 1) as f64;
        let delta = x - mean;
        mean += delta / k;
        let delta2 = x - mean;
        m2 += delta * delta2;
    }
    Ok((mean, m2))
}

// [cite:clinical_trials_statistics]
