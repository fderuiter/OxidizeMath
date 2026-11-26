//! Historical empirical, parametric Gaussian, and Cornish-Fisher `VaR` and `CVaR` kernels.

use super::RiskError;
use crate::pure_math::statistics::copula::{Probability, inverse_standard_normal};
use std::f64::consts::TAU;

const VOLATILITY_EPS: f64 = 1e-15;

/// Validates that `confidence` lies strictly inside `(0.5, 1.0)`.
#[verified_engine::verified]
pub fn validate_confidence(confidence: f64) -> Result<f64, RiskError> {
    if !confidence.is_finite() || confidence <= 0.5 || confidence >= 1.0 {
        return Err(RiskError::InvalidConfidence { value: confidence });
    }
    Ok(confidence)
}

fn validate_and_sort_returns<'a>(
    returns: &[f64],
    scratch: &'a mut [f64],
) -> Result<&'a [f64], RiskError> {
    if returns.is_empty() {
        return Err(RiskError::EmptySeries);
    }
    if scratch.len() < returns.len() {
        return Err(RiskError::ScratchBufferTooSmall {
            required: returns.len(),
            actual: scratch.len(),
        });
    }
    let work = &mut scratch[..returns.len()];
    for (dst, &val) in work.iter_mut().zip(returns.iter()) {
        if !val.is_finite() {
            return Err(RiskError::NonFiniteInput);
        }
        *dst = val;
    }
    work.sort_unstable_by(f64::total_cmp);
    Ok(work)
}

fn standard_normal_quantile_and_pdf(confidence: f64) -> Result<(f64, f64), RiskError> {
    let alpha = validate_confidence(confidence)?;
    let prob =
        Probability::new(alpha).map_err(|_| RiskError::InvalidConfidence { value: alpha })?;
    let z =
        inverse_standard_normal(prob).map_err(|_| RiskError::InvalidConfidence { value: alpha })?;
    let pdf = (-0.5 * z * z).exp() / TAU.sqrt();
    Ok((z, pdf))
}

struct StandardizedMoments {
    mean: f64,
    sample_std: f64,
    skewness: f64,
    excess_kurtosis: f64,
}

fn compute_moments(returns: &[f64], min_samples: usize) -> Result<StandardizedMoments, RiskError> {
    if returns.is_empty() {
        return Err(RiskError::EmptySeries);
    }
    if returns.len() < min_samples {
        return Err(RiskError::InsufficientSamples {
            required: min_samples,
            actual: returns.len(),
        });
    }
    let n = returns.len() as f64;
    let mut sum = 0.0;
    for &r in returns {
        if !r.is_finite() {
            return Err(RiskError::NonFiniteInput);
        }
        sum += r;
    }
    let mean = sum / n;
    let mut m2 = 0.0;
    let mut m3 = 0.0;
    let mut m4 = 0.0;
    for &r in returns {
        let d = r - mean;
        let d2 = d * d;
        m2 += d2;
        m3 += d2 * d;
        m4 += d2 * d2;
    }
    let sample_var = m2 / (n - 1.0);
    if sample_var <= VOLATILITY_EPS * VOLATILITY_EPS {
        return Err(RiskError::ZeroVolatility);
    }
    let pop_var = m2 / n;
    let pop_std = pop_var.sqrt();
    let skewness = (m3 / n) / (pop_var * pop_std);
    let excess_kurtosis = (m4 / n) / (pop_var * pop_var) - 3.0;
    Ok(StandardizedMoments {
        mean,
        sample_std: sample_var.sqrt(),
        skewness,
        excess_kurtosis,
    })
}

/// Computes empirical historical Value-at-Risk (`VaR`) as a positive left-tail loss fraction
/// for `confidence` $\alpha \in (0.5, 1.0)$ using caller-owned `scratch` (`scratch.len() >= returns.len()`).
#[verified_engine::verified]
pub fn historical_var(
    returns: &[f64],
    confidence: f64,
    scratch: &mut [f64],
) -> Result<f64, RiskError> {
    let alpha = validate_confidence(confidence)?;
    let sorted = validate_and_sort_returns(returns, scratch)?;
    let tail_mass = (1.0 - alpha) * (sorted.len() as f64);
    let idx = (tail_mass.floor() as usize).min(sorted.len() - 1);
    Ok(-sorted[idx])
}

/// Computes coherent empirical Expected Shortfall (`CVaR`, Acerbi & Tasche 2002) for
/// `confidence` $\alpha \in (0.5, 1.0)$ using caller-owned `scratch` (`scratch.len() >= returns.len()`).
///
/// Satisfies $\text{CVaR}_\alpha \ge \text{VaR}_\alpha$ and monotonicity in $\alpha$ for all finite sample sizes.
#[verified_engine::verified]
pub fn historical_cvar(
    returns: &[f64],
    confidence: f64,
    scratch: &mut [f64],
) -> Result<f64, RiskError> {
    let alpha = validate_confidence(confidence)?;
    let sorted = validate_and_sort_returns(returns, scratch)?;
    let tail_prob = 1.0 - alpha;
    let n = sorted.len() as f64;
    let k = ((tail_prob * n).floor() as usize).min(sorted.len() - 1);
    let mut loss_sum = 0.0;
    for &r in &sorted[..k] {
        loss_sum += -r;
    }
    let fractional_weight = tail_prob - (k as f64) / n;
    let boundary_loss = -sorted[k];
    Ok((loss_sum / n + fractional_weight * boundary_loss) / tail_prob)
}

/// Computes parametric Gaussian Value-at-Risk (`VaR`) $\text{VaR}_\alpha = -\mu + \sigma \Phi^{-1}(\alpha)$
/// for `confidence` $\alpha \in (0.5, 1.0)$.
#[verified_engine::verified]
pub fn gaussian_var(returns: &[f64], confidence: f64) -> Result<f64, RiskError> {
    let (z, _) = standard_normal_quantile_and_pdf(confidence)?;
    let mom = compute_moments(returns, 2)?;
    Ok(-mom.mean + mom.sample_std * z)
}

/// Computes parametric Gaussian Expected Shortfall (`CVaR`)
/// $\text{CVaR}_\alpha = -\mu + \sigma \frac{\phi(\Phi^{-1}(\alpha))}{1 - \alpha}$
/// for `confidence` $\alpha \in (0.5, 1.0)$.
#[verified_engine::verified]
pub fn gaussian_cvar(returns: &[f64], confidence: f64) -> Result<f64, RiskError> {
    let alpha = validate_confidence(confidence)?;
    let (_z, pdf) = standard_normal_quantile_and_pdf(alpha)?;
    let mom = compute_moments(returns, 2)?;
    let tail_expectation = pdf / (1.0 - alpha);
    Ok(-mom.mean + mom.sample_std * tail_expectation)
}

fn cornish_fisher_loss_quantile(z: f64, skew: f64, excess_kurt: f64) -> f64 {
    let z2 = z * z;
    let z3 = z2 * z;
    z - ((z2 - 1.0) / 6.0) * skew + ((z3 - 3.0 * z) / 24.0) * excess_kurt
        - ((2.0 * z3 - 5.0 * z) / 36.0) * skew * skew
}

/// Computes 4th-order Cornish-Fisher semi-parametric Value-at-Risk (`VaR`) incorporating
/// sample skewness and excess kurtosis for `confidence` $\alpha \in (0.5, 1.0)$.
#[verified_engine::verified]
pub fn cornish_fisher_var(returns: &[f64], confidence: f64) -> Result<f64, RiskError> {
    let (z, _) = standard_normal_quantile_and_pdf(confidence)?;
    let mom = compute_moments(returns, 3)?;
    let w_cf = cornish_fisher_loss_quantile(z, mom.skewness, mom.excess_kurtosis);
    Ok(-mom.mean + mom.sample_std * w_cf)
}

/// Computes 4th-order Cornish-Fisher Expected Shortfall (`CVaR`) via exact Hermite tail
/// integration over $u \in [\alpha, 1)$ for `confidence` $\alpha \in (0.5, 1.0)$.
#[verified_engine::verified]
pub fn cornish_fisher_cvar(returns: &[f64], confidence: f64) -> Result<f64, RiskError> {
    let alpha = validate_confidence(confidence)?;
    let (z, pdf) = standard_normal_quantile_and_pdf(alpha)?;
    let mom = compute_moments(returns, 3)?;
    let w_cf = cornish_fisher_loss_quantile(z, mom.skewness, mom.excess_kurtosis);
    let var_cf = -mom.mean + mom.sample_std * w_cf;
    let m1 = pdf / (1.0 - alpha);
    let z2 = z * z;
    let cf_tail_multiplier = m1
        * (1.0 - (z / 6.0) * mom.skewness + ((z2 - 1.0) / 24.0) * mom.excess_kurtosis
            - ((2.0 * z2 - 1.0) / 36.0) * mom.skewness * mom.skewness);
    let cvar_raw = -mom.mean + mom.sample_std * cf_tail_multiplier;
    Ok(cvar_raw.max(var_cf))
}

// [cite:clinical_trials_statistics]
