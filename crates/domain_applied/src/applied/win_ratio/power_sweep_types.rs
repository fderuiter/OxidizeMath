//! Configuration and report types for Monte Carlo composite win-ratio power sweeps.

use crate::applied::clinical_trials::types::ClinicalTrialError;

use super::censoring::TierCount;
use super::u_statistic::UnmatchedWinRatioSummary;

/// Maximum per-arm sample size permitted in WebAssembly and interactive sweeps.
pub const MAX_ARM_SIZE: usize = 500;

/// Maximum Monte Carlo replicate count permitted per sample-size grid step.
pub const MAX_SWEEP_REPLICATES: usize = 500;

/// Maximum number of sample-size grid steps in a single power sweep.
pub const MAX_SWEEP_STEPS: usize = 16;

/// Configuration for a seeded Monte Carlo composite win-ratio power sweep across arm sizes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PowerSweepConfig {
    /// Minimum sample size per arm ($n_{\min} \ge 2$).
    pub n_min: usize,
    /// Maximum sample size per arm ($n_{\min} \le n_{\max} \le 500$).
    pub n_max: usize,
    /// Number of evenly spaced sample-size grid points ($1 \le \text{steps} \le 16$).
    pub n_steps: usize,
    /// Number of simulated two-arm trials per grid point ($1 \le R \le 500$).
    pub replicates: usize,
    /// Administrative follow-up censoring horizon $c > 0$.
    pub censor_horizon: f64,
    /// Exponential random loss-to-follow-up hazard rate ($\ge 0$).
    pub dropout_rate: f64,
    /// Two-sided significance level $\alpha \in (0, 1)$ (e.g., `0.05`).
    pub sig_alpha: f64,
}

impl Default for PowerSweepConfig {
    fn default() -> Self {
        Self {
            n_min: 30,
            n_max: 150,
            n_steps: 5,
            replicates: 60,
            censor_horizon: 5.0,
            dropout_rate: 0.05,
            sig_alpha: 0.05,
        }
    }
}

impl PowerSweepConfig {
    /// Validates all sample-size, replicate, censoring, and significance bounds.
    #[verified_engine::verified]
    pub fn validate(&self) -> Result<(), ClinicalTrialError> {
        if self.n_min < 2 || self.n_max < self.n_min || self.n_max > MAX_ARM_SIZE {
            return Err(ClinicalTrialError::InvalidData(format!(
                "Arm sizes must satisfy 2 <= n_min <= n_max <= {MAX_ARM_SIZE}, got {}..={}",
                self.n_min, self.n_max
            )));
        }
        if self.n_steps == 0 || self.n_steps > MAX_SWEEP_STEPS {
            return Err(ClinicalTrialError::InvalidData(format!(
                "n_steps must be in 1..={MAX_SWEEP_STEPS}, got {}",
                self.n_steps
            )));
        }
        if self.replicates == 0 || self.replicates > MAX_SWEEP_REPLICATES {
            return Err(ClinicalTrialError::InvalidData(format!(
                "replicates must be in 1..={MAX_SWEEP_REPLICATES}, got {}",
                self.replicates
            )));
        }
        if !self.censor_horizon.is_finite() || self.censor_horizon <= 0.0 {
            return Err(ClinicalTrialError::InvalidData(
                "censor_horizon must be positive and finite".to_string(),
            ));
        }
        if !self.dropout_rate.is_finite() || self.dropout_rate < 0.0 {
            return Err(ClinicalTrialError::InvalidData(
                "dropout_rate must be non-negative and finite".to_string(),
            ));
        }
        if !self.sig_alpha.is_finite() || self.sig_alpha <= 0.0 || self.sig_alpha >= 1.0 {
            return Err(ClinicalTrialError::InvalidData(
                "sig_alpha must be finite in (0, 1)".to_string(),
            ));
        }
        debug_assert!(self.n_min <= self.n_max && self.n_steps <= MAX_SWEEP_STEPS);
        Ok(())
    }

    /// Computes the `step_idx`-th sample size per arm in `0..self.n_steps`.
    #[verified_engine::verified]
    #[must_use]
    pub fn sample_size_at(&self, step_idx: usize) -> usize {
        if self.n_steps <= 1 {
            return self.n_max;
        }
        let span = self.n_max - self.n_min;
        let numer = span * step_idx + (self.n_steps - 1) / 2;
        let out = self.n_min + numer / (self.n_steps - 1);
        debug_assert!(out >= self.n_min);
        out
    }
}

/// Aggregated Monte Carlo power and tier waterfall metrics at a single per-arm sample size.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct SampleSizePowerPoint {
    /// Subjects per arm ($N$).
    pub n_per_arm: usize,
    /// Empirical rejection rate ($\Pr(p < \alpha)$) across Monte Carlo replicates.
    pub empirical_power: f64,
    /// Mean finite win ratio across replicates (geometric mean on log-scale).
    pub mean_win_ratio: f64,
    /// Mean Tier 1 (fatal) win fraction across replicates.
    pub tier1_win_fraction: f64,
    /// Mean Tier 1 (fatal) loss fraction across replicates.
    pub tier1_loss_fraction: f64,
    /// Mean Tier 2 (non-fatal) win fraction across replicates.
    pub tier2_win_fraction: f64,
    /// Mean Tier 2 (non-fatal) loss fraction across replicates.
    pub tier2_loss_fraction: f64,
    /// Mean overall tie fraction across replicates.
    pub tie_fraction: f64,
}

/// Output report from a deterministic `BivariateWeibullModel` power sweep.
#[derive(Debug, Clone, PartialEq)]
pub struct PowerSweepReport {
    /// Theoretical uncensored win ratio $PR_W = \theta^{-\alpha}$.
    pub theoretical_uncensored_wr: f64,
    /// Representative trial U-statistic summary at `n_max`.
    pub representative_summary: UnmatchedWinRatioSummary,
    /// Representative trial tier waterfall (`[Fatal, Non-Fatal]`) at `n_max`.
    pub representative_tiers: [TierCount; 2],
    /// Power-by-$N$ curve points ordered from `n_min` to `n_max`.
    pub points: Vec<SampleSizePowerPoint>,
}

// [cite:win_ratio]
