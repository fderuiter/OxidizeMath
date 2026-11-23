//! Nonparametric percentile and jackknife-accelerated (`BCa`) bootstrap workspace.
//!
//! Preallocates resample, replicate, scratch, and leave-one-out jackknife buffers once at
//! initialization so [`BootstrapWorkspace`] executes with zero dynamic heap allocations inside
//! `VerifiedAllocator` blocks.
//!
//! ## Example
//!
//! ```rust
//! # use pure_math::__doctest_facade as math_explorer;
//! use math_explorer::prelude::OxidizeRng;
//! use math_explorer::pure_math::statistics::bootstrap::{BootstrapWorkspace, sample_mean};
//!
//! fn main() -> math_explorer::error::Result<()> {
//!     let data = [1.2, 1.5, 1.8, 2.1, 2.9, 4.4, 6.2, 9.5];
//!     let mut ws = BootstrapWorkspace::new(data.len(), 1000)?;
//!     let mut rng = OxidizeRng::new(42);
//!     let bca = ws.bca_ci(&data, 1000, 0.95, &mut rng, sample_mean)?;
//!     assert!(bca.ci_lower < bca.point_estimate && bca.point_estimate < bca.ci_upper);
//!     Ok(())
//! }
//! ```

mod bca;

use crate::pure_math::statistics::monte_carlo::workspace::draw_bounded_index;
use crate::pure_math::statistics::monte_carlo::{MonteCarloError, two_sided_normal_z};
use bca::{build_bca_estimate, sorted_quantile};
use oxidize_core::rng::OxidizeRng;

/// Arithmetic mean of a non-empty slice without heap allocation.
#[inline]
#[verified_engine::verified]
pub fn sample_mean(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    data.iter().sum::<f64>() / (data.len() as f64)
}

/// Exact sample median computed in-place using `scratch` with zero heap allocations.
#[inline]
#[verified_engine::verified]
pub fn sample_median_in_place(data: &[f64], scratch: &mut [f64]) -> f64 {
    let n = data.len();
    if n == 0 || scratch.len() < n {
        return 0.0;
    }
    let buf = &mut scratch[..n];
    buf.copy_from_slice(data);
    buf.sort_unstable_by(f64::total_cmp);
    if n % 2 == 1 {
        buf[n / 2]
    } else {
        0.5 * (buf[n / 2 - 1] + buf[n / 2])
    }
}

/// Nonparametric bootstrap confidence interval method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BootstrapMethod {
    /// Equal-tail empirical percentile interval $[\hat{\theta}^*_{\alpha/2}, \hat{\theta}^*_{1-\alpha/2}]$.
    Percentile,
    /// Efron's bias-corrected and jackknife-accelerated (`BCa`) interval.
    BCa,
}

/// Calibrated nonparametric bootstrap estimate and confidence interval summary.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BootstrapEstimate {
    /// Observed statistic $\hat{\theta} = T(x_1, \dots, x_n)$ on the original dataset.
    pub point_estimate: f64,
    /// Mean of the bootstrap replicate distribution $\bar{\theta}^*$.
    pub bootstrap_mean: f64,
    /// Bootstrap standard error across the `replicates` resamples.
    pub standard_error: f64,
    /// Estimated bootstrap bias $\bar{\theta}^* - \hat{\theta}$.
    pub bias: f64,
    /// Lower confidence interval endpoint.
    pub ci_lower: f64,
    /// Upper confidence interval endpoint.
    pub ci_upper: f64,
    /// Nominal two-sided confidence level in `(0.0, 1.0)`.
    pub confidence_level: f64,
    /// Number of bootstrap resamples evaluated.
    pub replicates: usize,
    /// Bias-correction constant $\hat{z}_0$ (`0.0` for [`BootstrapMethod::Percentile`]).
    pub bias_correction_z0: f64,
    /// Jackknife acceleration constant $\hat{a}$ (`0.0` for [`BootstrapMethod::Percentile`]).
    pub acceleration: f64,
    /// Interval construction method used.
    pub method: BootstrapMethod,
}

/// Paired [`BootstrapMethod::Percentile`] and [`BootstrapMethod::BCa`] estimates from a shared draw.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BootstrapComparison {
    /// Equal-tail percentile interval over the bootstrap distribution.
    pub percentile: BootstrapEstimate,
    /// Bias-corrected and jackknife-accelerated (`BCa`) interval over the same bootstrap distribution.
    pub bca: BootstrapEstimate,
}

/// Preallocated workspace for zero-allocation percentile and `BCa` nonparametric bootstrap inference.
#[derive(Debug, Clone)]
pub struct BootstrapWorkspace {
    max_data_len: usize,
    max_replicates: usize,
    last_replicate_count: usize,
    resample_buf: Vec<f64>,
    replicates_buf: Vec<f64>,
    jackknife_buf: Vec<f64>,
    scratch_buf: Vec<f64>,
}

impl BootstrapWorkspace {
    /// Preallocates resample, replicate, scratch, and jackknife buffers.
    #[verified_engine::verified]
    pub fn new(max_data_len: usize, max_replicates: usize) -> Result<Self, MonteCarloError> {
        if max_data_len < 2 {
            return Err(MonteCarloError::InsufficientData {
                required: 2,
                actual: max_data_len,
            });
        }
        if max_replicates == 0 {
            return Err(MonteCarloError::ZeroReplicates);
        }
        Ok(Self {
            max_data_len,
            max_replicates,
            last_replicate_count: 0,
            resample_buf: vec![0.0; max_data_len],
            replicates_buf: vec![0.0; max_replicates],
            jackknife_buf: vec![0.0; max_data_len],
            scratch_buf: vec![0.0; max_data_len],
        })
    }

    /// Maximum dataset length supported without reallocation.
    #[inline]
    #[verified_engine::verified]
    pub fn max_data_len(&self) -> usize {
        self.max_data_len
    }

    /// Maximum number of bootstrap replicates supported without reallocation.
    #[inline]
    #[verified_engine::verified]
    pub fn max_replicates(&self) -> usize {
        self.max_replicates
    }

    /// Sorted slice of bootstrap replicates generated by the most recent estimation call.
    #[inline]
    #[verified_engine::verified]
    pub fn last_replicates(&self) -> &[f64] {
        &self.replicates_buf[..self.last_replicate_count]
    }

    #[verified_engine::verified]
    fn validate_inputs(
        &self,
        data: &[f64],
        n_replicates: usize,
        confidence_level: f64,
    ) -> Result<f64, MonteCarloError> {
        if data.len() < 2 {
            return Err(MonteCarloError::InsufficientData {
                required: 2,
                actual: data.len(),
            });
        }
        if data.len() > self.max_data_len {
            return Err(MonteCarloError::BufferLengthMismatch {
                expected: self.max_data_len,
                actual: data.len(),
            });
        }
        if n_replicates == 0 || n_replicates > self.max_replicates {
            return Err(MonteCarloError::ZeroReplicates);
        }
        for &val in data {
            if !val.is_finite() {
                return Err(MonteCarloError::NonFiniteEvaluation);
            }
        }
        two_sided_normal_z(confidence_level)
    }

    #[verified_engine::verified]
    fn populate_sorted_replicates<F>(
        &mut self,
        data: &[f64],
        n_replicates: usize,
        rng: &mut OxidizeRng,
        statistic: &mut F,
    ) -> Result<(f64, f64, f64), MonteCarloError>
    where
        F: FnMut(&[f64], &mut [f64]) -> f64,
    {
        let n = data.len();
        let point_est = statistic(data, &mut self.scratch_buf[..n]);
        if !point_est.is_finite() {
            return Err(MonteCarloError::NonFiniteEvaluation);
        }
        for rep_idx in 0..n_replicates {
            for slot in &mut self.resample_buf[..n] {
                *slot = data[draw_bounded_index(rng, n)];
            }
            let star = statistic(&self.resample_buf[..n], &mut self.scratch_buf[..n]);
            if !star.is_finite() {
                return Err(MonteCarloError::NonFiniteEvaluation);
            }
            self.replicates_buf[rep_idx] = star;
        }
        let reps = &mut self.replicates_buf[..n_replicates];
        reps.sort_unstable_by(f64::total_cmp);
        self.last_replicate_count = n_replicates;

        let mean = sample_mean(reps);
        let se = if n_replicates > 1 {
            let ss: f64 = reps.iter().map(|&v| (v - mean) * (v - mean)).sum();
            (ss / ((n_replicates - 1) as f64)).sqrt()
        } else {
            0.0
        };
        Ok((point_est, mean, se))
    }

    #[verified_engine::verified]
    fn compute_jackknife_acceleration<F>(
        &mut self,
        data: &[f64],
        statistic: &mut F,
    ) -> Result<f64, MonteCarloError>
    where
        F: FnMut(&[f64], &mut [f64]) -> f64,
    {
        let n = data.len();
        let loo_len = n - 1;
        for leave_out in 0..n {
            self.resample_buf[..leave_out].copy_from_slice(&data[..leave_out]);
            self.resample_buf[leave_out..loo_len].copy_from_slice(&data[leave_out + 1..]);
            let val = statistic(
                &self.resample_buf[..loo_len],
                &mut self.scratch_buf[..loo_len],
            );
            if !val.is_finite() {
                return Err(MonteCarloError::NonFiniteEvaluation);
            }
            self.jackknife_buf[leave_out] = val;
        }
        let jack = &self.jackknife_buf[..n];
        let jack_mean = sample_mean(jack);
        let mut sum_d2 = 0.0;
        let mut sum_d3 = 0.0;
        for &theta_i in jack {
            let d = jack_mean - theta_i;
            let d2 = d * d;
            sum_d2 += d2;
            sum_d3 += d2 * d;
        }
        let denom = 6.0 * sum_d2.powf(1.5);
        Ok(if denom > 1e-30 { sum_d3 / denom } else { 0.0 })
    }

    /// Equal-tail nonparametric percentile bootstrap interval.
    #[verified_engine::verified]
    pub fn percentile_ci<F>(
        &mut self,
        data: &[f64],
        n_replicates: usize,
        confidence_level: f64,
        rng: &mut OxidizeRng,
        mut statistic: F,
    ) -> Result<BootstrapEstimate, MonteCarloError>
    where
        F: FnMut(&[f64]) -> f64,
    {
        self.percentile_ci_with_scratch(
            data,
            n_replicates,
            confidence_level,
            rng,
            |xs, _scratch| statistic(xs),
        )
    }

    /// Equal-tail percentile bootstrap interval passing a preallocated `&mut [f64]` scratch buffer.
    #[verified_engine::verified]
    pub fn percentile_ci_with_scratch<F>(
        &mut self,
        data: &[f64],
        n_replicates: usize,
        confidence_level: f64,
        rng: &mut OxidizeRng,
        mut statistic: F,
    ) -> Result<BootstrapEstimate, MonteCarloError>
    where
        F: FnMut(&[f64], &mut [f64]) -> f64,
    {
        let _ = self.validate_inputs(data, n_replicates, confidence_level)?;
        let (point_estimate, bootstrap_mean, standard_error) =
            self.populate_sorted_replicates(data, n_replicates, rng, &mut statistic)?;
        let alpha_half = 0.5 * (1.0 - confidence_level);
        let reps = &self.replicates_buf[..n_replicates];
        Ok(BootstrapEstimate {
            point_estimate,
            bootstrap_mean,
            standard_error,
            bias: bootstrap_mean - point_estimate,
            ci_lower: sorted_quantile(reps, alpha_half),
            ci_upper: sorted_quantile(reps, 1.0 - alpha_half),
            confidence_level,
            replicates: n_replicates,
            bias_correction_z0: 0.0,
            acceleration: 0.0,
            method: BootstrapMethod::Percentile,
        })
    }

    /// Efron's bias-corrected and jackknife-accelerated (`BCa`) bootstrap confidence interval.
    #[verified_engine::verified]
    pub fn bca_ci<F>(
        &mut self,
        data: &[f64],
        n_replicates: usize,
        confidence_level: f64,
        rng: &mut OxidizeRng,
        mut statistic: F,
    ) -> Result<BootstrapEstimate, MonteCarloError>
    where
        F: FnMut(&[f64]) -> f64,
    {
        self.bca_ci_with_scratch(data, n_replicates, confidence_level, rng, |xs, _scratch| {
            statistic(xs)
        })
    }

    /// `BCa` bootstrap confidence interval passing a preallocated `&mut [f64]` scratch buffer.
    #[verified_engine::verified]
    pub fn bca_ci_with_scratch<F>(
        &mut self,
        data: &[f64],
        n_replicates: usize,
        confidence_level: f64,
        rng: &mut OxidizeRng,
        statistic: F,
    ) -> Result<BootstrapEstimate, MonteCarloError>
    where
        F: FnMut(&[f64], &mut [f64]) -> f64,
    {
        Ok(self
            .compare_intervals_with_scratch(data, n_replicates, confidence_level, rng, statistic)?
            .bca)
    }

    /// Computes paired `Percentile` and `BCa` intervals from the same bootstrap draw.
    #[verified_engine::verified]
    pub fn compare_intervals<F>(
        &mut self,
        data: &[f64],
        n_replicates: usize,
        confidence_level: f64,
        rng: &mut OxidizeRng,
        mut statistic: F,
    ) -> Result<BootstrapComparison, MonteCarloError>
    where
        F: FnMut(&[f64]) -> f64,
    {
        self.compare_intervals_with_scratch(
            data,
            n_replicates,
            confidence_level,
            rng,
            |xs, _scratch| statistic(xs),
        )
    }

    /// Computes paired `Percentile` and `BCa` intervals using a preallocated scratch slice.
    #[verified_engine::verified]
    pub fn compare_intervals_with_scratch<F>(
        &mut self,
        data: &[f64],
        n_replicates: usize,
        confidence_level: f64,
        rng: &mut OxidizeRng,
        mut statistic: F,
    ) -> Result<BootstrapComparison, MonteCarloError>
    where
        F: FnMut(&[f64], &mut [f64]) -> f64,
    {
        let z_upper = self.validate_inputs(data, n_replicates, confidence_level)?;
        let (point_estimate, bootstrap_mean, standard_error) =
            self.populate_sorted_replicates(data, n_replicates, rng, &mut statistic)?;
        let acceleration = self.compute_jackknife_acceleration(data, &mut statistic)?;
        let reps = &self.replicates_buf[..n_replicates];
        let alpha_half = 0.5 * (1.0 - confidence_level);
        let bias = bootstrap_mean - point_estimate;

        let percentile = BootstrapEstimate {
            point_estimate,
            bootstrap_mean,
            standard_error,
            bias,
            ci_lower: sorted_quantile(reps, alpha_half),
            ci_upper: sorted_quantile(reps, 1.0 - alpha_half),
            confidence_level,
            replicates: n_replicates,
            bias_correction_z0: 0.0,
            acceleration: 0.0,
            method: BootstrapMethod::Percentile,
        };
        let bca = build_bca_estimate(
            reps,
            point_estimate,
            bootstrap_mean,
            standard_error,
            confidence_level,
            z_upper,
            acceleration,
        );
        Ok(BootstrapComparison { percentile, bca })
    }
}

#[cfg(test)]
mod tests;

// [cite:clinical_trials_statistics]
