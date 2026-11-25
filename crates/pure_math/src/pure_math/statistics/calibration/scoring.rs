//! Proper scoring rules for binary, categorical, Gaussian, and ensemble probability forecasts.
// [cite:calibration]

use std::f64::consts::PI;

use math_commons::error::{MathError, MathResult};
use math_commons::math_kernel::types::Dimension;
use math_commons::primitives::{PositiveFloat, UnitInterval};
use statrs::distribution::{Continuous, ContinuousCDF, Normal};

use super::diagnostics::{BinningStrategy, reliability_curve};

/// Default symmetric probability clipping bound $\epsilon = 10^{-15}$ for logarithmic scoring rules.
pub const DEFAULT_LOG_LOSS_EPSILON: f64 = 1e-15;

/// Murphy (1973) three-component decomposition of the quadratic Brier score.
///
/// Satisfies the algebraic partition $\text{BS} = \text{REL} - \text{RES} + \text{UNC}$, where
/// lower `reliability` indicates better calibration, higher `resolution` indicates stronger case
/// discrimination, and `uncertainty` reflects marginal outcome entropy $\bar{y}(1 - \bar{y})$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BrierDecomposition {
    /// Mean squared probability error $\frac{1}{N}\sum_{i=1}^N (p_i - y_i)^2$.
    pub brier_score: f64,
    /// Weighted mean squared calibration error $\frac{1}{N}\sum_{k=1}^K n_k (f_k - \bar{y}_k)^2$.
    pub reliability: f64,
    /// Weighted variance of conditional event rates $\frac{1}{N}\sum_{k=1}^K n_k (\bar{y}_k - \bar{y})^2$.
    pub resolution: f64,
    /// Marginal climatological variance $\bar{y}(1 - \bar{y})$ of the binary outcomes.
    pub uncertainty: f64,
}

impl BrierDecomposition {
    /// Returns the absolute algebraic residual $|\text{REL} - \text{RES} + \text{UNC} - \text{BS}|$.
    #[verified_engine::verified]
    pub fn identity_residual(&self) -> f64 {
        (self.reliability - self.resolution + self.uncertainty - self.brier_score).abs()
    }
}

pub(crate) fn validate_binary_inputs(probs: &[UnitInterval], outcomes: &[bool]) -> MathResult<()> {
    if probs.len() != outcomes.len() {
        return Err(MathError::DimensionMismatch {
            expected: Dimension(outcomes.len()),
            actual: Dimension(probs.len()),
        });
    }
    if probs.is_empty() {
        return Err(MathError::NumericalError {
            reason: "Calibration input slices must be non-empty",
        });
    }
    Ok(())
}

fn validate_clip_epsilon(epsilon: f64) -> MathResult<()> {
    if !epsilon.is_finite() || !(f64::EPSILON..0.5).contains(&epsilon) {
        return Err(MathError::NumericalError {
            reason: "Clipping epsilon must be finite and within [f64::EPSILON, 0.5)",
        });
    }
    Ok(())
}

fn sum_brier_squared_errors(probs: &[UnitInterval], outcomes: &[bool]) -> f64 {
    let mut sum = 0.0;
    for (p, &y) in probs.iter().zip(outcomes.iter()) {
        let target = if y { 1.0 } else { 0.0 };
        let diff = p.value() - target;
        sum += diff * diff;
    }
    sum
}

/// Computes the binary Brier score $\frac{1}{N}\sum_{i=1}^N (p_i - y_i)^2$ with zero heap allocation.
#[verified_engine::verified]
pub fn brier_score(probs: &[UnitInterval], outcomes: &[bool]) -> MathResult<f64> {
    validate_binary_inputs(probs, outcomes)?;
    Ok(sum_brier_squared_errors(probs, outcomes) / (probs.len() as f64))
}

fn accumulate_murphy_groups(sorted_pairs: &[(f64, bool)], base_rate: f64) -> (f64, f64) {
    let n = sorted_pairs.len() as f64;
    let mut rel_sum = 0.0;
    let mut res_sum = 0.0;
    let mut idx = 0;
    while idx < sorted_pairs.len() {
        let f_k = sorted_pairs[idx].0;
        let mut count = 0_usize;
        let mut pos = 0_usize;
        while idx < sorted_pairs.len() && sorted_pairs[idx].0.to_bits() == f_k.to_bits() {
            count += 1;
            if sorted_pairs[idx].1 {
                pos += 1;
            }
            idx += 1;
        }
        let n_k = count as f64;
        let o_k = (pos as f64) / n_k;
        let rel_diff = f_k - o_k;
        let res_diff = o_k - base_rate;
        rel_sum += n_k * rel_diff * rel_diff;
        res_sum += n_k * res_diff * res_diff;
    }
    (rel_sum / n, res_sum / n)
}

/// Computes the exact Murphy (1973) decomposition grouped over distinct forecast values.
///
/// Guarantees `brier_score = reliability - resolution + uncertainty` within machine precision
/// (`< math_commons::registry::TOLERANCE_STANDARD`) for every valid input dataset.
#[verified_engine::verified]
pub fn murphy_decomposition(
    probs: &[UnitInterval],
    outcomes: &[bool],
) -> MathResult<BrierDecomposition> {
    validate_binary_inputs(probs, outcomes)?;
    let n = probs.len() as f64;
    let bs = sum_brier_squared_errors(probs, outcomes) / n;
    let pos_count = outcomes.iter().filter(|&&y| y).count() as f64;
    let base_rate = pos_count / n;
    let uncertainty = base_rate * (1.0 - base_rate);

    let mut pairs: Vec<(f64, bool)> = probs
        .iter()
        .zip(outcomes.iter())
        .map(|(p, &y)| (p.value(), y))
        .collect();
    pairs.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));

    let (reliability, resolution) = accumulate_murphy_groups(&pairs, base_rate);
    Ok(BrierDecomposition {
        brier_score: bs,
        reliability,
        resolution,
        uncertainty,
    })
}

/// Computes a binned Murphy decomposition by quantizing continuous forecasts to empirical bin means.
///
/// For continuous neural or statistical risk scores where every raw forecast $p_i$ is unique,
/// grouping into `num_bins` via [`BinningStrategy`] yields interpretable bin-level `reliability`
/// and `resolution` while preserving `brier_score = reliability - resolution + uncertainty` for
/// the bin-quantized forecast $\bar{p}_{b(i)}$.
#[verified_engine::verified]
pub fn murphy_decomposition_binned(
    probs: &[UnitInterval],
    outcomes: &[bool],
    num_bins: usize,
    strategy: BinningStrategy,
) -> MathResult<BrierDecomposition> {
    let bins = reliability_curve(probs, outcomes, num_bins, strategy)?;
    let n = probs.len() as f64;
    let pos_count = outcomes.iter().filter(|&&y| y).count() as f64;
    let base_rate = pos_count / n;
    let uncertainty = base_rate * (1.0 - base_rate);

    let mut brier_binned = 0.0;
    let mut reliability = 0.0;
    let mut resolution = 0.0;
    for bin in &bins {
        if bin.count == 0 {
            continue;
        }
        let weight = (bin.count as f64) / n;
        let p_b = bin.mean_confidence;
        let y_b = bin.empirical_accuracy;
        let rel_diff = p_b - y_b;
        let res_diff = y_b - base_rate;
        brier_binned += weight * (y_b * (p_b - 1.0) * (p_b - 1.0) + (1.0 - y_b) * p_b * p_b);
        reliability += weight * rel_diff * rel_diff;
        resolution += weight * res_diff * res_diff;
    }
    Ok(BrierDecomposition {
        brier_score: brier_binned,
        reliability,
        resolution,
        uncertainty,
    })
}

/// Computes binary dataset log-loss with default symmetric clipping [`DEFAULT_LOG_LOSS_EPSILON`].
#[verified_engine::verified]
pub fn log_loss(probs: &[UnitInterval], outcomes: &[bool]) -> MathResult<f64> {
    binary_log_loss(probs, outcomes, DEFAULT_LOG_LOSS_EPSILON)
}

/// Computes binary dataset log-loss with symmetric `[epsilon, 1.0 - epsilon]` probability clipping
/// and explicit negative-class complement terms $-(y \ln p_c + (1 - y) \ln(1 - p_c))$.
#[verified_engine::verified]
pub fn binary_log_loss(probs: &[UnitInterval], outcomes: &[bool], epsilon: f64) -> MathResult<f64> {
    validate_binary_inputs(probs, outcomes)?;
    validate_clip_epsilon(epsilon)?;
    let upper = 1.0 - epsilon;
    let mut total = 0.0;
    for (p, &y) in probs.iter().zip(outcomes.iter()) {
        let p_c = p.value().clamp(epsilon, upper);
        let q_c = p.complement().clamp(epsilon, upper);
        total -= if y { p_c.ln() } else { q_c.ln() };
    }
    Ok(total / (probs.len() as f64))
}

fn validate_multiclass_row(row: &[UnitInterval], target: usize) -> MathResult<()> {
    if target >= row.len() {
        return Err(MathError::NumericalError {
            reason: "Target class index out of bounds for multiclass probability row",
        });
    }
    let row_sum: f64 = row.iter().map(|p| p.value()).sum();
    if (row_sum - 1.0).abs() > 1e-3 {
        return Err(MathError::NumericalError {
            reason: "Multiclass probability row must sum to 1.0",
        });
    }
    Ok(())
}

/// Computes multiclass dataset log-loss over row-major `probs_flat` (`targets.len() * num_classes`)
/// with symmetric `[epsilon, 1.0 - epsilon]` clipping.
#[verified_engine::verified]
pub fn multiclass_log_loss(
    probs_flat: &[UnitInterval],
    num_classes: usize,
    targets: &[usize],
    epsilon: f64,
) -> MathResult<f64> {
    if num_classes < 2 {
        return Err(MathError::NumericalError {
            reason: "multiclass_log_loss requires at least 2 classes",
        });
    }
    if targets.is_empty() {
        return Err(MathError::NumericalError {
            reason: "multiclass_log_loss requires non-empty targets",
        });
    }
    let expected_len = targets
        .len()
        .checked_mul(num_classes)
        .ok_or(MathError::NumericalError {
            reason: "Sample count * num_classes overflowed usize",
        })?;
    if probs_flat.len() != expected_len {
        return Err(MathError::DimensionMismatch {
            expected: Dimension(expected_len),
            actual: Dimension(probs_flat.len()),
        });
    }
    validate_clip_epsilon(epsilon)?;
    let upper = 1.0 - epsilon;
    let mut total = 0.0;
    for (row_idx, &target) in targets.iter().enumerate() {
        let row = &probs_flat[row_idx * num_classes..(row_idx + 1) * num_classes];
        validate_multiclass_row(row, target)?;
        let p_c = row[target].value().clamp(epsilon, upper);
        total -= p_c.ln();
    }
    Ok(total / (targets.len() as f64))
}

fn gaussian_crps_with_normal(
    std_normal: &Normal,
    mean: f64,
    std_dev: PositiveFloat,
    observation: f64,
) -> MathResult<f64> {
    if !mean.is_finite() || !observation.is_finite() {
        return Err(MathError::NumericalError {
            reason: "Gaussian CRPS mean and observation must be finite",
        });
    }
    let sigma = std_dev.value();
    let z = (observation - mean) / sigma;
    let cdf = std_normal.cdf(z);
    let pdf = std_normal.pdf(z);
    let inv_sqrt_pi = 1.0 / PI.sqrt();
    let score = sigma * (z * (2.0 * cdf - 1.0) + 2.0 * pdf - inv_sqrt_pi);
    Ok(score.max(0.0))
}

/// Computes the analytical Continuous Ranked Probability Score (`CRPS`) for a Gaussian predictive
/// distribution $\mathcal{N}(\mu, \sigma^2)$ evaluated at scalar `observation` $y$.
///
/// Uses the closed-form identity (Gneiting & Raftery, 2007):
/// $\text{CRPS}(\mathcal{N}(\mu, \sigma^2), y) = \sigma [z(2\Phi(z) - 1) + 2\phi(z) - \pi^{-1/2}]$,
/// where $z = (y - \mu)/\sigma$.
#[verified_engine::verified]
pub fn gaussian_crps(mean: f64, std_dev: PositiveFloat, observation: f64) -> MathResult<f64> {
    let std_normal = Normal::new(0.0, 1.0).map_err(|_| MathError::NumericalError {
        reason: "Failed to construct standard normal distribution",
    })?;
    gaussian_crps_with_normal(&std_normal, mean, std_dev, observation)
}

/// Computes the mean analytical Gaussian `CRPS` across a dataset of predictive means and scales.
#[verified_engine::verified]
pub fn dataset_gaussian_crps(
    means: &[f64],
    std_devs: &[PositiveFloat],
    observations: &[f64],
) -> MathResult<f64> {
    if means.len() != observations.len() {
        return Err(MathError::DimensionMismatch {
            expected: Dimension(observations.len()),
            actual: Dimension(means.len()),
        });
    }
    if std_devs.len() != observations.len() {
        return Err(MathError::DimensionMismatch {
            expected: Dimension(observations.len()),
            actual: Dimension(std_devs.len()),
        });
    }
    if observations.is_empty() {
        return Err(MathError::NumericalError {
            reason: "dataset_gaussian_crps requires non-empty slices",
        });
    }
    let std_normal = Normal::new(0.0, 1.0).map_err(|_| MathError::NumericalError {
        reason: "Failed to construct standard normal distribution",
    })?;
    let mut sum = 0.0;
    for i in 0..observations.len() {
        sum += gaussian_crps_with_normal(&std_normal, means[i], std_devs[i], observations[i])?;
    }
    Ok(sum / (observations.len() as f64))
}

fn eval_sorted_ensemble_crps(sorted_samples: &[f64], observation: f64) -> MathResult<f64> {
    let m = sorted_samples.len();
    let base = sorted_samples[0];
    if !base.is_finite() {
        return Err(MathError::NumericalError {
            reason: "Ensemble sample at index 0 is non-finite",
        });
    }
    let mut abs_err_sum = (base - observation).abs();
    let mut spread_sum = 0.0;
    let m_i64 = m as f64;
    for (j, &x_j) in sorted_samples.iter().enumerate().skip(1) {
        if !x_j.is_finite() || x_j < sorted_samples[j - 1] {
            return Err(MathError::NumericalError {
                reason: "ensemble_crps_sorted requires finite non-decreasing samples",
            });
        }
        abs_err_sum += (x_j - observation).abs();
        let coeff = 2.0 * (j as f64) + 1.0 - m_i64;
        spread_sum += coeff * (x_j - base);
    }
    let mae = abs_err_sum / m_i64;
    let dispersion = spread_sum / (m_i64 * m_i64);
    Ok((mae - dispersion).max(0.0))
}

/// Computes the empirical ensemble `CRPS` for a pre-sorted slice `sorted_samples` in $O(M)$ time
/// with zero heap allocations.
///
/// Evaluates $\frac{1}{M}\sum_{j=0}^{M-1} |x_j - y| - \frac{1}{M^2}\sum_{j=0}^{M-1}(2j + 1 - M)(x_j - x_0)$,
/// using shift-invariant offsets $(x_j - x_0)$ to prevent catastrophic cancellation.
#[verified_engine::verified]
pub fn ensemble_crps_sorted(sorted_samples: &[f64], observation: f64) -> MathResult<f64> {
    if sorted_samples.is_empty() {
        return Err(MathError::NumericalError {
            reason: "ensemble_crps_sorted requires at least one ensemble member",
        });
    }
    if !observation.is_finite() {
        return Err(MathError::NumericalError {
            reason: "Observation must be finite for ensemble CRPS",
        });
    }
    eval_sorted_ensemble_crps(sorted_samples, observation)
}
