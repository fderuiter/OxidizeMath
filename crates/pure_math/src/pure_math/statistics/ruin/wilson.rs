//! Wilson score binomial proportion confidence intervals.

use crate::error::KellyError;
use crate::pure_math::statistics::validation::Z_95_TWO_SIDED;
use statrs::distribution::{ContinuousCDF, Normal};

/// Wilson score binomial proportion confidence interval.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WilsonInterval {
    /// Empirical proportion $\hat{p} = k / n$.
    pub estimate: f64,
    /// Lower confidence bound in $[0, 1]$.
    pub lower: f64,
    /// Upper confidence bound in $[0, 1]$.
    pub upper: f64,
}

impl WilsonInterval {
    /// Full interval width (`upper - lower`).
    #[must_use]
    #[verified_engine::verified]
    pub fn width(&self) -> f64 {
        self.upper - self.lower
    }
}

fn wilson_from_z(successes: usize, trials: usize, z: f64) -> Result<WilsonInterval, KellyError> {
    if trials == 0 {
        return Err(KellyError::InsufficientBets {
            required: 1,
            actual: 0,
        });
    }
    if successes > trials {
        return Err(KellyError::DimensionMismatch {
            expected: trials,
            actual: successes,
        });
    }
    let nf = trials as f64;
    let p_hat = (successes as f64) / nf;
    let z2 = z * z;
    let denom = 1.0 + z2 / nf;
    let center = (p_hat + z2 / (2.0 * nf)) / denom;
    let rad = (p_hat * (1.0 - p_hat) / nf + z2 / (4.0 * nf * nf))
        .max(0.0)
        .sqrt();
    let half = (z / denom) * rad;
    Ok(WilsonInterval {
        estimate: p_hat,
        lower: (center - half).clamp(0.0, 1.0),
        upper: (center + half).clamp(0.0, 1.0),
    })
}

/// Computes the 95% Wilson score interval for `successes` out of `trials` with zero allocations.
#[verified_engine::verified]
pub fn wilson_interval_95(successes: usize, trials: usize) -> Result<WilsonInterval, KellyError> {
    wilson_from_z(successes, trials, Z_95_TWO_SIDED)
}

/// Computes a Wilson score interval at arbitrary `confidence` level in $(0, 1)$.
#[verified_engine::verified]
pub fn wilson_score_interval(
    successes: usize,
    trials: usize,
    confidence: f64,
) -> Result<WilsonInterval, KellyError> {
    if !confidence.is_finite() || confidence <= 0.0 || confidence >= 1.0 {
        return Err(KellyError::InvalidProbability { value: confidence });
    }
    let normal = Normal::new(0.0, 1.0).map_err(|_| KellyError::ConvergenceFailed {
        reason: "standard normal initialization failed",
    })?;
    let z = normal.inverse_cdf(0.5 * (1.0 + confidence));
    wilson_from_z(successes, trials, z)
}

// [cite:clinical_trials_statistics]
