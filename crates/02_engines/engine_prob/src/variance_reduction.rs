//! `#![no_std]` variance-reduction estimators: antithetic variates, control variates, and SNIS.

use crate::error::ProbError;
use crate::math::{box_muller_pair, Z_95_TWO_SIDED};
use crate::workspace::SolverWorkspace;
use oxidize_core::rng::OxidizeRng;

#[path = "bernoulli_antithetic.rs"]
mod bernoulli_antithetic;

pub use bernoulli_antithetic::{
    simulate_correlated_bernoulli_antithetic, simulate_orthant_antithetic_col_major,
    workspace_floats_for_correlated_bernoulli, CorrelatedBernoulliSummary,
};

/// Monte Carlo expectation summary with sample variance, standard error, and 95% normal bounds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MonteCarloEstimate {
    /// Sample mean estimate.
    pub mean: f64,
    /// Standard error of `mean`.
    pub std_error: f64,
    /// Variance across independent sample units (or antithetic pairs).
    pub variance: f64,
    /// Lower 95% asymptotic confidence bound (`mean - 1.96 * std_error`).
    pub lower_95: f64,
    /// Upper 95% asymptotic confidence bound (`mean + 1.96 * std_error`).
    pub upper_95: f64,
    /// Total integrand or model evaluations executed.
    pub n_evaluations: usize,
}

/// Optimal linear control-variate summary $\hat{\mu}_{\text{CV}} = \bar{Y} - \hat{\beta}^*(\bar{C} - \mathbb{E}\[C\])$.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ControlVariateEstimate {
    /// Control-variate adjusted mean estimate.
    pub mean: f64,
    /// Standard error of the adjusted mean.
    pub std_error: f64,
    /// Estimated optimal regression slope $\hat{\beta}^* = \widehat{\mathrm{Cov}}(Y, C) / \widehat{\mathrm{Var}}(C)$.
    pub beta_hat: f64,
    /// Unadjusted IID sample mean $\bar{Y}$.
    pub raw_mean: f64,
    /// Unadjusted IID standard error of $\bar{Y}$.
    pub raw_std_error: f64,
    /// Ratio of controlled variance to raw variance ($\le 1$ when correlated).
    pub variance_reduction_ratio: f64,
    /// Lower 95% confidence bound on `mean`.
    pub lower_95: f64,
    /// Upper 95% confidence bound on `mean`.
    pub upper_95: f64,
    /// Number of paired `(target, control)` samples processed.
    pub n_samples: usize,
}

/// Self-normalized importance sampling (SNIS) summary with Kish effective sample size.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImportanceSamplingEstimate {
    /// Self-normalized weighted expectation $\sum w_i h_i / \sum w_i$.
    pub mean: f64,
    /// Asymptotic delta-method standard error $\sqrt{\sum w_i^2 (h_i - \hat{\mu})^2} / \sum w_i$.
    pub std_error: f64,
    /// Kish effective sample size $(\sum w_i)^2 / \sum w_i^2 \in [1, n]$.
    pub effective_sample_size: f64,
    /// Sum of unnormalized non-negative importance weights $\sum w_i$.
    pub weight_sum: f64,
    /// Lower 95% asymptotic confidence bound on `mean`.
    pub lower_95: f64,
    /// Upper 95% asymptotic confidence bound on `mean`.
    pub upper_95: f64,
    /// Total proposal draws evaluated.
    pub n_samples: usize,
}

/// Evaluates $\mathbb{E}[f(Z)]$ for $Z \sim \mathcal{N}(0, I_d)$ using `n_pairs` antithetic Gaussian
/// pairs $(+Z_k, -Z_k)$ over [`SolverWorkspace`] (`2 * dim` floats required).
pub fn estimate_antithetic_normal<F: FnMut(&[f64]) -> f64>(
    dim: usize,
    n_pairs: usize,
    rng: &mut OxidizeRng,
    workspace: &mut SolverWorkspace<'_>,
    mut f: F,
) -> Result<MonteCarloEstimate, ProbError> {
    if dim == 0 {
        return Err(ProbError::InvalidDimension {
            dim: 0,
            max_dim: usize::MAX,
        });
    }
    if n_pairs == 0 {
        return Err(ProbError::InvalidCount {
            name: "n_pairs",
            value: 0,
        });
    }
    let (z_pos, mut rem) = workspace.split_floats(dim)?;
    let (z_neg, _) = rem.split_floats(dim)?;

    let mut mean = 0.0;
    let mut m2 = 0.0;
    for pair_idx in 0..n_pairs {
        let mut j = 0;
        while j < dim {
            let (g0, g1) = box_muller_pair(rng);
            z_pos[j] = g0;
            z_neg[j] = -g0;
            if j + 1 < dim {
                z_pos[j + 1] = g1;
                z_neg[j + 1] = -g1;
            }
            j += 2;
        }
        let y_pos = f(z_pos);
        let y_neg = f(z_neg);
        if !y_pos.is_finite() || !y_neg.is_finite() {
            return Err(ProbError::NonFiniteInput {
                context: "antithetic normal integrand",
            });
        }
        let pair_avg = 0.5 * (y_pos + y_neg);
        let k = (pair_idx + 1) as f64;
        let delta = pair_avg - mean;
        mean += delta / k;
        m2 += delta * (pair_avg - mean);
    }

    let variance = if n_pairs > 1 {
        m2 / ((n_pairs - 1) as f64)
    } else {
        0.0
    };
    let std_error = libm::sqrt(variance / (n_pairs as f64));
    let half = Z_95_TWO_SIDED * std_error;
    Ok(MonteCarloEstimate {
        mean,
        std_error,
        variance,
        lower_95: mean - half,
        upper_95: mean + half,
        n_evaluations: 2 * n_pairs,
    })
}

/// Evaluates $\mathbb{E}[f(U)]$ for $U \sim \mathrm{Uniform}(0, 1)^d$ using `n_pairs` antithetic
/// pairs $(U_k, 1 - U_k)$ over [`SolverWorkspace`] (`2 * dim` floats required).
pub fn estimate_antithetic_uniform<F: FnMut(&[f64]) -> f64>(
    dim: usize,
    n_pairs: usize,
    rng: &mut OxidizeRng,
    workspace: &mut SolverWorkspace<'_>,
    mut f: F,
) -> Result<MonteCarloEstimate, ProbError> {
    if dim == 0 {
        return Err(ProbError::InvalidDimension {
            dim: 0,
            max_dim: usize::MAX,
        });
    }
    if n_pairs == 0 {
        return Err(ProbError::InvalidCount {
            name: "n_pairs",
            value: 0,
        });
    }
    let (u_pos, mut rem) = workspace.split_floats(dim)?;
    let (u_anti, _) = rem.split_floats(dim)?;

    let mut mean = 0.0;
    let mut m2 = 0.0;
    for pair_idx in 0..n_pairs {
        for j in 0..dim {
            let u = rng.next_f64_open01();
            u_pos[j] = u;
            u_anti[j] = 1.0 - u;
        }
        let y_pos = f(u_pos);
        let y_anti = f(u_anti);
        if !y_pos.is_finite() || !y_anti.is_finite() {
            return Err(ProbError::NonFiniteInput {
                context: "antithetic uniform integrand",
            });
        }
        let pair_avg = 0.5 * (y_pos + y_anti);
        let k = (pair_idx + 1) as f64;
        let delta = pair_avg - mean;
        mean += delta / k;
        m2 += delta * (pair_avg - mean);
    }

    let variance = if n_pairs > 1 {
        m2 / ((n_pairs - 1) as f64)
    } else {
        0.0
    };
    let std_error = libm::sqrt(variance / (n_pairs as f64));
    let half = Z_95_TWO_SIDED * std_error;
    Ok(MonteCarloEstimate {
        mean,
        std_error,
        variance,
        lower_95: mean - half,
        upper_95: mean + half,
        n_evaluations: 2 * n_pairs,
    })
}

fn paired_means_and_moments(
    targets: &[f64],
    controls: &[f64],
) -> Result<(f64, f64, f64, f64, f64), ProbError> {
    let n = targets.len();
    let (mut sum_y, mut sum_c) = (0.0, 0.0);
    for i in 0..n {
        if !targets[i].is_finite() || !controls[i].is_finite() {
            return Err(ProbError::NonFiniteInput {
                context: "control variate slices",
            });
        }
        sum_y += targets[i];
        sum_c += controls[i];
    }
    let nf = n as f64;
    let (mean_y, mean_c) = (sum_y / nf, sum_c / nf);
    let (mut s_yy, mut s_cc, mut s_yc) = (0.0, 0.0, 0.0);
    for i in 0..n {
        let dy = targets[i] - mean_y;
        let dc = controls[i] - mean_c;
        s_yy += dy * dy;
        s_cc += dc * dc;
        s_yc += dy * dc;
    }
    Ok((mean_y, mean_c, s_yy, s_cc, s_yc))
}

/// Computes optimal linear control-variate expectation from paired `targets` and `controls` slices
/// given exact analytical `control_expectation = E[C]`.
pub fn control_variate_from_slices(
    targets: &[f64],
    controls: &[f64],
    control_expectation: f64,
) -> Result<ControlVariateEstimate, ProbError> {
    let n = targets.len();
    if n == 0 || controls.len() != n {
        return Err(ProbError::EmptySample);
    }
    if !control_expectation.is_finite() {
        return Err(ProbError::NonFiniteInput {
            context: "control_expectation",
        });
    }
    let (mean_y, mean_c, s_yy, s_cc, s_yc) = paired_means_and_moments(targets, controls)?;
    let nf = n as f64;
    let beta_hat = if n > 1 && s_cc > 1e-30 {
        s_yc / s_cc
    } else {
        0.0
    };
    let adj_mean = mean_y - beta_hat * (mean_c - control_expectation);

    let raw_var = if n > 1 { s_yy / ((n - 1) as f64) } else { 0.0 };
    let mut s_rr = 0.0;
    for i in 0..n {
        let r_i = targets[i] - beta_hat * (controls[i] - control_expectation) - adj_mean;
        s_rr += r_i * r_i;
    }
    let dof = if n > 2 && beta_hat != 0.0 {
        (n - 2) as f64
    } else if n > 1 {
        (n - 1) as f64
    } else {
        1.0
    };
    let cv_var = if n > 1 { s_rr / dof } else { 0.0 };
    let raw_se = libm::sqrt(raw_var / nf);
    let std_error = libm::sqrt(cv_var / nf);
    let vr_ratio = if raw_var > 1e-30 {
        (cv_var / raw_var).max(0.0)
    } else {
        1.0
    };
    let half = Z_95_TWO_SIDED * std_error;

    Ok(ControlVariateEstimate {
        mean: adj_mean,
        std_error,
        beta_hat,
        raw_mean: mean_y,
        raw_std_error: raw_se,
        variance_reduction_ratio: vr_ratio,
        lower_95: adj_mean - half,
        upper_95: adj_mean + half,
        n_samples: n,
    })
}

/// Samples `n_samples` `(target, control)` pairs into [`SolverWorkspace`] (`2 * n_samples` floats)
/// and returns the optimal linear control-variate estimate.
pub fn estimate_control_variate<F: FnMut(&mut OxidizeRng) -> (f64, f64)>(
    n_samples: usize,
    control_expectation: f64,
    rng: &mut OxidizeRng,
    workspace: &mut SolverWorkspace<'_>,
    mut sampler: F,
) -> Result<ControlVariateEstimate, ProbError> {
    if n_samples == 0 {
        return Err(ProbError::InvalidCount {
            name: "n_samples",
            value: 0,
        });
    }
    let (targets, mut rem) = workspace.split_floats(n_samples)?;
    let (controls, _) = rem.split_floats(n_samples)?;
    for i in 0..n_samples {
        let (y, c) = sampler(rng);
        targets[i] = y;
        controls[i] = c;
    }
    control_variate_from_slices(targets, controls, control_expectation)
}

/// Computes the self-normalized importance sampling (SNIS) estimate, Kish ESS, and delta-method
/// standard error from pre-evaluated `values` and non-negative unnormalized `weights`.
pub fn self_normalized_is_from_slices(
    values: &[f64],
    weights: &[f64],
) -> Result<ImportanceSamplingEstimate, ProbError> {
    let n = values.len();
    if n == 0 || weights.len() != n {
        return Err(ProbError::EmptySample);
    }
    let mut w_sum = 0.0;
    let mut w2_sum = 0.0;
    let mut wh_sum = 0.0;
    for i in 0..n {
        let h = values[i];
        let w = weights[i];
        if !h.is_finite() || !w.is_finite() {
            return Err(ProbError::NonFiniteInput {
                context: "importance sampling inputs",
            });
        }
        if w < 0.0 {
            return Err(ProbError::ZeroEffectiveWeight);
        }
        w_sum += w;
        w2_sum += w * w;
        wh_sum += w * h;
    }
    if w_sum <= 0.0 || w2_sum <= 0.0 {
        return Err(ProbError::ZeroEffectiveWeight);
    }
    let mean = wh_sum / w_sum;
    let ess = (w_sum * w_sum) / w2_sum;
    let mut num_var = 0.0;
    for i in 0..n {
        let diff = values[i] - mean;
        let w = weights[i];
        num_var += (w * w) * (diff * diff);
    }
    let std_error = if n > 1 {
        libm::sqrt(num_var) / w_sum
    } else {
        0.0
    };
    let half = Z_95_TWO_SIDED * std_error;
    Ok(ImportanceSamplingEstimate {
        mean,
        std_error,
        effective_sample_size: ess,
        weight_sum: w_sum,
        lower_95: mean - half,
        upper_95: mean + half,
        n_samples: n,
    })
}

/// Draws `n_samples` `(value, unnormalized_weight)` pairs into [`SolverWorkspace`] (`2 * n_samples`
/// floats) and computes the self-normalized importance sampling estimate.
pub fn estimate_self_normalized_is<F: FnMut(&mut OxidizeRng) -> (f64, f64)>(
    n_samples: usize,
    rng: &mut OxidizeRng,
    workspace: &mut SolverWorkspace<'_>,
    mut proposal_sampler: F,
) -> Result<ImportanceSamplingEstimate, ProbError> {
    if n_samples == 0 {
        return Err(ProbError::InvalidCount {
            name: "n_samples",
            value: 0,
        });
    }
    let (values, mut rem) = workspace.split_floats(n_samples)?;
    let (weights, _) = rem.split_floats(n_samples)?;
    for i in 0..n_samples {
        let (v, w) = proposal_sampler(rng);
        values[i] = v;
        weights[i] = w;
    }
    self_normalized_is_from_slices(values, weights)
}
