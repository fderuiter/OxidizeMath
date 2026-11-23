//! Zero-allocation nonparametric percentile and Efron BCa bootstrap estimators over [`SolverWorkspace`].

use crate::error::ProbError;
use crate::math::{
    mean_and_sample_variance, sort_total_f64, sorted_quantile, standard_normal_cdf,
    standard_normal_inv_cdf,
};
use crate::workspace::SolverWorkspace;
use oxidize_core::rng::OxidizeRng;

/// Nonparametric bootstrap interval construction method.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BootstrapMethod {
    /// Equal-tailed empirical percentile interval $[\hat{\theta}^*_{\alpha/2}, \hat{\theta}^*_{1-\alpha/2}]$.
    #[default]
    Percentile,
    /// Efron (1987) bias-corrected and accelerated ($\mathrm{BC}_a$) interval with jackknife $\hat{a}$.
    BCa,
}

/// Configuration parameters for nonparametric bootstrap resampling.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BootstrapConfig {
    /// Number of bootstrap resamples $B \ge 2$.
    pub n_resamples: usize,
    /// Two-sided nominal confidence level in $(0, 1)$ (e.g., `0.95`).
    pub confidence_level: f64,
    /// Interval construction method (`Percentile` or `BCa`).
    pub method: BootstrapMethod,
}

impl BootstrapConfig {
    /// Creates a `Percentile` bootstrap configuration with `n_resamples` and `confidence_level`.
    #[inline]
    #[must_use]
    pub const fn new(n_resamples: usize, confidence_level: f64) -> Self {
        Self {
            n_resamples,
            confidence_level,
            method: BootstrapMethod::Percentile,
        }
    }

    /// Overrides the bootstrap interval construction method (`Percentile` or `BCa`).
    #[inline]
    #[must_use]
    pub const fn with_method(mut self, method: BootstrapMethod) -> Self {
        self.method = method;
        self
    }
}

impl Default for BootstrapConfig {
    fn default() -> Self {
        Self::new(500, 0.95)
    }
}

/// Nonparametric bootstrap confidence interval and diagnostic summary.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BootstrapInterval {
    /// Statistic evaluated on the original unresampled dataset $\hat{\theta}$.
    pub point_estimate: f64,
    /// Lower confidence bound at level `confidence`.
    pub lower: f64,
    /// Upper confidence bound at level `confidence`.
    pub upper: f64,
    /// Bootstrap standard error across `n_resamples` replicates.
    pub std_error: f64,
    /// Nominal two-sided confidence level in $(0, 1)$.
    pub confidence: f64,
    /// Median bias-correction $z_0 = \Phi^{-1}(\hat{\mathbb{P}}(\hat{\theta}^* < \hat{\theta}))$ (`0.0` for percentile).
    pub bias_z0: f64,
    /// Leave-one-out jackknife acceleration $\hat{a}$ (`0.0` for percentile or degenerate variance).
    pub acceleration: f64,
}

/// Required `f64` slots in [`SolverWorkspace`] for scalar percentile bootstrap (`n_resamples + sample_len`).
#[inline]
#[must_use]
pub const fn percentile_workspace_floats(sample_len: usize, n_resamples: usize) -> usize {
    n_resamples + sample_len
}

/// Required `f64` slots in [`SolverWorkspace`] for scalar BCa bootstrap (`n_resamples + 2 * sample_len`).
#[inline]
#[must_use]
pub const fn bca_workspace_floats(sample_len: usize, n_resamples: usize) -> usize {
    n_resamples + 2 * sample_len
}

/// Required `usize` slots in [`SolverWorkspace`] for index-based bootstrap (`sample_len`).
#[inline]
#[must_use]
pub const fn bootstrap_workspace_indices(sample_len: usize) -> usize {
    sample_len
}

fn validate_bootstrap_args(
    sample_len: usize,
    n_resamples: usize,
    confidence: f64,
) -> Result<(), ProbError> {
    if sample_len == 0 {
        return Err(ProbError::EmptySample);
    }
    if n_resamples == 0 {
        return Err(ProbError::InvalidCount {
            name: "n_resamples",
            value: 0,
        });
    }
    if !confidence.is_finite() || confidence <= 0.0 || confidence >= 1.0 {
        return Err(ProbError::InvalidConfidence { confidence });
    }
    Ok(())
}

fn s3_over_s2_accel(jack_vals: &[f64], mean_jack: f64) -> f64 {
    let mut s2 = 0.0;
    let mut s3 = 0.0;
    for &jv in jack_vals {
        let diff = mean_jack - jv;
        let d2 = diff * diff;
        s2 += d2;
        s3 += d2 * diff;
    }
    if s2 <= 1e-30 {
        0.0
    } else {
        s3 / (6.0 * libm::pow(s2, 1.5))
    }
}

fn compute_jackknife_acceleration_indices<S: FnMut(&[usize]) -> f64>(
    sample_len: usize,
    idx_buf: &mut [usize],
    jack_vals: &mut [f64],
    statistic: &mut S,
) -> Result<f64, ProbError> {
    if sample_len <= 1 {
        return Ok(0.0);
    }
    let mut sum = 0.0;
    for (leave_out, jack_slot) in jack_vals.iter_mut().enumerate().take(sample_len) {
        let mut pos = 0;
        for k in 0..sample_len {
            if k != leave_out {
                idx_buf[pos] = k;
                pos += 1;
            }
        }
        let val = statistic(&idx_buf[..sample_len - 1]);
        if !val.is_finite() {
            return Err(ProbError::NonFiniteInput {
                context: "jackknife statistic",
            });
        }
        *jack_slot = val;
        sum += val;
    }
    Ok(s3_over_s2_accel(
        &jack_vals[..sample_len],
        sum / (sample_len as f64),
    ))
}

fn compute_jackknife_acceleration_floats<S: FnMut(&[f64]) -> f64>(
    sample: &[f64],
    draw_buf: &mut [f64],
    jack_vals: &mut [f64],
    statistic: &mut S,
) -> Result<f64, ProbError> {
    let n = sample.len();
    if n <= 1 {
        return Ok(0.0);
    }
    let mut sum_j = 0.0;
    for (leave_out, jack_slot) in jack_vals.iter_mut().enumerate().take(n) {
        let mut pos = 0;
        for (k, &v) in sample.iter().enumerate() {
            if k != leave_out {
                draw_buf[pos] = v;
                pos += 1;
            }
        }
        let jv = statistic(&draw_buf[..n - 1]);
        if !jv.is_finite() {
            return Err(ProbError::NonFiniteInput {
                context: "jackknife statistic",
            });
        }
        *jack_slot = jv;
        sum_j += jv;
    }
    Ok(s3_over_s2_accel(&jack_vals[..n], sum_j / (n as f64)))
}

fn finalize_sorted_interval(
    reps: &mut [f64],
    point_estimate: f64,
    confidence: f64,
    accel: Option<f64>,
) -> Result<BootstrapInterval, ProbError> {
    let (_, rep_var) = mean_and_sample_variance(reps)?;
    let std_error = libm::sqrt(rep_var);
    sort_total_f64(reps);
    if reps[reps.len() - 1] - reps[0] <= f64::EPSILON * (1.0 + libm::fabs(point_estimate)) {
        return Ok(BootstrapInterval {
            point_estimate,
            lower: point_estimate,
            upper: point_estimate,
            std_error: 0.0,
            confidence,
            bias_z0: 0.0,
            acceleration: 0.0,
        });
    }
    if let Some(a) = accel {
        let (lower, upper, bias_z0) = bca_adjusted_quantiles(reps, point_estimate, confidence, a);
        Ok(BootstrapInterval {
            point_estimate,
            lower,
            upper,
            std_error,
            confidence,
            bias_z0,
            acceleration: a,
        })
    } else {
        let alpha_half = 0.5 * (1.0 - confidence);
        Ok(BootstrapInterval {
            point_estimate,
            lower: sorted_quantile(reps, alpha_half),
            upper: sorted_quantile(reps, 1.0 - alpha_half),
            std_error,
            confidence,
            bias_z0: 0.0,
            acceleration: 0.0,
        })
    }
}

fn bca_adjusted_quantiles(
    sorted_reps: &[f64],
    point_estimate: f64,
    confidence: f64,
    acceleration: f64,
) -> (f64, f64, f64) {
    let b = sorted_reps.len() as f64;
    let mut less = 0_usize;
    let mut equal = 0_usize;
    for &rep in sorted_reps {
        if rep < point_estimate {
            less += 1;
        } else if rep == point_estimate {
            equal += 1;
        }
    }
    let min_p = 0.5 / b;
    let prop = (((less as f64) + 0.5 * (equal as f64)) / b).clamp(min_p, 1.0 - min_p);
    let z0 = standard_normal_inv_cdf(prop);

    let alpha_half = 0.5 * (1.0 - confidence);
    let z_hi = standard_normal_inv_cdf(1.0 - alpha_half);
    let z_lo = -z_hi;

    let map_level = |z_alpha: f64| -> f64 {
        let w = z0 + z_alpha;
        let denom = 1.0 - acceleration * w;
        if denom <= 1e-12 {
            1.0
        } else {
            standard_normal_cdf(z0 + w / denom)
        }
    };

    let q_lo = sorted_quantile(sorted_reps, map_level(z_lo));
    let q_hi = sorted_quantile(sorted_reps, map_level(z_hi));
    (q_lo.min(q_hi), q_lo.max(q_hi), z0)
}

/// Runs zero-allocation index-based nonparametric bootstrap (`Percentile` or `BCa`) over a sample
/// of length `sample_len` using [`SolverWorkspace`] (`n_resamples + sample_len` floats and `sample_len` indices).
pub fn bootstrap_indices<S: FnMut(&[usize]) -> f64>(
    sample_len: usize,
    n_resamples: usize,
    confidence: f64,
    method: BootstrapMethod,
    rng: &mut OxidizeRng,
    workspace: &mut SolverWorkspace<'_>,
    mut statistic: S,
) -> Result<BootstrapInterval, ProbError> {
    validate_bootstrap_args(sample_len, n_resamples, confidence)?;
    let (reps, idx_buf, mut rem) = workspace.split_parts(n_resamples, sample_len)?;
    for (k, slot) in idx_buf.iter_mut().enumerate() {
        *slot = k;
    }
    let point_estimate = statistic(idx_buf);
    if !point_estimate.is_finite() {
        return Err(ProbError::NonFiniteInput {
            context: "bootstrap point_estimate",
        });
    }
    for rep_slot in reps.iter_mut() {
        for slot in idx_buf.iter_mut() {
            *slot = rng.next_index(sample_len);
        }
        let val = statistic(idx_buf);
        if !val.is_finite() {
            return Err(ProbError::NonFiniteInput {
                context: "bootstrap replicate",
            });
        }
        *rep_slot = val;
    }
    let accel = if method == BootstrapMethod::BCa {
        let (jack_vals, _) = rem.split_floats(sample_len)?;
        Some(compute_jackknife_acceleration_indices(
            sample_len,
            idx_buf,
            jack_vals,
            &mut statistic,
        )?)
    } else {
        None
    };
    finalize_sorted_interval(reps, point_estimate, confidence, accel)
}

/// Zero-allocation percentile bootstrap for a scalar slice `sample: &[f64]` using [`SolverWorkspace`]
/// (`n_resamples + sample.len()` floats required; zero `usize` indices required).
pub fn percentile_bootstrap<S: FnMut(&[f64]) -> f64>(
    sample: &[f64],
    n_resamples: usize,
    confidence: f64,
    rng: &mut OxidizeRng,
    workspace: &mut SolverWorkspace<'_>,
    mut statistic: S,
) -> Result<BootstrapInterval, ProbError> {
    let n = sample.len();
    validate_bootstrap_args(n, n_resamples, confidence)?;
    if sample.iter().any(|x| !x.is_finite()) {
        return Err(ProbError::NonFiniteInput {
            context: "bootstrap sample",
        });
    }
    let (reps, mut rem) = workspace.split_floats(n_resamples)?;
    let (draw_buf, _) = rem.split_floats(n)?;
    let point_estimate = statistic(sample);
    if !point_estimate.is_finite() {
        return Err(ProbError::NonFiniteInput {
            context: "bootstrap point_estimate",
        });
    }
    for rep_slot in reps.iter_mut() {
        for slot in draw_buf.iter_mut() {
            *slot = sample[rng.next_index(n)];
        }
        let val = statistic(draw_buf);
        if !val.is_finite() {
            return Err(ProbError::NonFiniteInput {
                context: "bootstrap replicate",
            });
        }
        *rep_slot = val;
    }
    finalize_sorted_interval(reps, point_estimate, confidence, None)
}

/// Zero-allocation Efron BCa bootstrap with leave-one-out jackknife acceleration for `sample: &[f64]`
/// using [`SolverWorkspace`] (`n_resamples + 2 * sample.len()` floats required; zero `usize` indices required).
pub fn bca_bootstrap<S: FnMut(&[f64]) -> f64>(
    sample: &[f64],
    n_resamples: usize,
    confidence: f64,
    rng: &mut OxidizeRng,
    workspace: &mut SolverWorkspace<'_>,
    mut statistic: S,
) -> Result<BootstrapInterval, ProbError> {
    let n = sample.len();
    validate_bootstrap_args(n, n_resamples, confidence)?;
    if sample.iter().any(|x| !x.is_finite()) {
        return Err(ProbError::NonFiniteInput {
            context: "bootstrap sample",
        });
    }
    let (reps, mut rem1) = workspace.split_floats(n_resamples)?;
    let (draw_buf, mut rem2) = rem1.split_floats(n)?;
    let (jack_vals, _) = rem2.split_floats(n)?;
    let point_estimate = statistic(sample);
    if !point_estimate.is_finite() {
        return Err(ProbError::NonFiniteInput {
            context: "bootstrap point_estimate",
        });
    }
    for rep_slot in reps.iter_mut() {
        for slot in draw_buf.iter_mut() {
            *slot = sample[rng.next_index(n)];
        }
        let val = statistic(draw_buf);
        if !val.is_finite() {
            return Err(ProbError::NonFiniteInput {
                context: "bootstrap replicate",
            });
        }
        *rep_slot = val;
    }
    let accel = compute_jackknife_acceleration_floats(sample, draw_buf, jack_vals, &mut statistic)?;
    finalize_sorted_interval(reps, point_estimate, confidence, Some(accel))
}

#[path = "bootstrap_shortcuts.rs"]
mod bootstrap_shortcuts;

pub use bootstrap_shortcuts::{
    bca_bootstrap_mean, bca_bootstrap_median, percentile_bootstrap_mean,
    percentile_bootstrap_median,
};
