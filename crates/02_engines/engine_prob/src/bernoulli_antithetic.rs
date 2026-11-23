//! Antithetic Cholesky correlated Bernoulli / Gaussian threshold indicator simulator.

use crate::error::ProbError;
use crate::math::{box_muller_pair, flatten_row_major};
use crate::workspace::SolverWorkspace;
use oxidize_core::rng::OxidizeRng;

/// Summary returned by [`simulate_correlated_bernoulli_antithetic`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CorrelatedBernoulliSummary {
    /// Antithetic Monte Carlo estimate of $\mathbb{P}(\forall i,\ Z_i \le z_i)$.
    pub joint_prob: f64,
    /// Standard error of `joint_prob` across antithetic pairs.
    pub joint_std_error: f64,
    /// Total joint indicator hits across all `2 * n_pairs` correlated paths.
    pub joint_hits: usize,
    /// Number of antithetic pairs simulated.
    pub n_pairs: usize,
    /// Total correlated paths simulated (`2 * n_pairs`).
    pub total_paths: usize,
}

/// Minimum `f64` scratch slots required in [`SolverWorkspace`] for `dim`-leg correlated Bernoulli simulation.
#[inline]
#[must_use]
pub const fn workspace_floats_for_correlated_bernoulli(dim: usize) -> usize {
    2 * dim
}

fn eval_orthant_hit(
    dim: usize,
    chol_lower: &[f64],
    thresholds_z: &[f64],
    z_indep: &[f64],
    sign: f64,
    z_corr: &mut [f64],
    marginal_hits: &mut [usize],
) -> bool {
    let mut all_hit = true;
    for i in 0..dim {
        let mut acc = 0.0;
        let row_offset = i * dim;
        for j in 0..=i {
            acc += chol_lower[row_offset + j] * (sign * z_indep[j]);
        }
        z_corr[i] = acc;
        if acc <= thresholds_z[i] {
            marginal_hits[i] += 1;
        } else {
            all_hit = false;
        }
    }
    all_hit
}

fn validate_bernoulli_inputs(
    chol_lower: &[f64],
    thresholds_z: &[f64],
    n_pairs: usize,
    marginal_hits_out: &[usize],
) -> Result<usize, ProbError> {
    let dim = thresholds_z.len();
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
    if chol_lower.len() < dim * dim || marginal_hits_out.len() < dim {
        return Err(ProbError::DegenerateMatrix {
            reason: "Cholesky factor or marginal_hits_out shorter than dimension",
        });
    }
    if thresholds_z.iter().any(|z| z.is_nan()) {
        return Err(ProbError::NonFiniteInput {
            context: "thresholds_z",
        });
    }
    Ok(dim)
}

fn fill_standard_normals(rng: &mut OxidizeRng, out: &mut [f64]) {
    let dim = out.len();
    let mut idx = 0;
    while idx < dim {
        let (g0, g1) = box_muller_pair(rng);
        out[idx] = g0;
        if idx + 1 < dim {
            out[idx + 1] = g1;
        }
        idx += 2;
    }
}

/// Simulates `n_pairs` antithetic Gaussian pairs $(+Z, -Z)$ mapped through row-major Cholesky
/// factor `chol_lower` (`dim * dim`) against upper thresholds `thresholds_z` (`dim`) with zero
/// heap allocations over [`SolverWorkspace`].
///
/// Updates `marginal_hits_out[i]` with individual marginal hit counts across all `2 * n_pairs`
/// paths and returns the joint hit probability and exact paired standard error.
pub fn simulate_correlated_bernoulli_antithetic(
    chol_lower: &[f64],
    thresholds_z: &[f64],
    n_pairs: usize,
    rng: &mut OxidizeRng,
    marginal_hits_out: &mut [usize],
    workspace: &mut SolverWorkspace<'_>,
) -> Result<CorrelatedBernoulliSummary, ProbError> {
    let dim = validate_bernoulli_inputs(chol_lower, thresholds_z, n_pairs, marginal_hits_out)?;
    let (z_indep, mut rem) = workspace.split_floats(dim)?;
    let (z_corr, _) = rem.split_floats(dim)?;
    marginal_hits_out[..dim].fill(0);

    let mut joint_hits: usize = 0;
    let (mut sum_pair_y, mut sum_pair_y2) = (0.0_f64, 0.0_f64);
    for _ in 0..n_pairs {
        fill_standard_normals(rng, z_indep);
        let hit_pos = eval_orthant_hit(
            dim,
            chol_lower,
            thresholds_z,
            z_indep,
            1.0,
            z_corr,
            marginal_hits_out,
        );
        let hit_neg = eval_orthant_hit(
            dim,
            chol_lower,
            thresholds_z,
            z_indep,
            -1.0,
            z_corr,
            marginal_hits_out,
        );
        let pair_hits = usize::from(hit_pos) + usize::from(hit_neg);
        joint_hits += pair_hits;
        let pair_y = 0.5 * (pair_hits as f64);
        sum_pair_y += pair_y;
        sum_pair_y2 += pair_y * pair_y;
    }

    let k = n_pairs as f64;
    let joint_prob = sum_pair_y / k;
    let joint_std_error = if n_pairs > 1 {
        let pair_var = ((sum_pair_y2 - k * joint_prob * joint_prob) / (k - 1.0)).max(0.0);
        libm::sqrt(pair_var / k)
    } else {
        0.0
    };
    Ok(CorrelatedBernoulliSummary {
        joint_prob,
        joint_std_error,
        joint_hits,
        n_pairs,
        total_paths: 2 * n_pairs,
    })
}

fn eval_col_major_pair(
    dim: usize,
    chol_col_major: &[f64],
    thresholds_z: &[f64],
    eps: &[f64],
) -> (bool, bool) {
    let mut pos_ok = true;
    let mut neg_ok = true;
    for i in 0..dim {
        let mut z_i = 0.0;
        for (j, &e_j) in eps.iter().enumerate().take(i + 1) {
            z_i += chol_col_major[flatten_row_major(j, i, dim)] * e_j;
        }
        let bound = thresholds_z[i];
        pos_ok &= z_i <= bound;
        neg_ok &= -z_i <= bound;
        if !pos_ok && !neg_ok {
            break;
        }
    }
    (pos_ok, neg_ok)
}

/// Simulates `n_pairs` antithetic Gaussian pairs $(+Z, -Z)$ mapped through a column-major
/// (`col * dim + row`) Cholesky factor `chol_col_major` against upper thresholds `thresholds_z`
/// using only `dim` scratch floats in [`SolverWorkspace`] and zero heap allocations.
pub fn simulate_orthant_antithetic_col_major(
    chol_col_major: &[f64],
    thresholds_z: &[f64],
    n_pairs: usize,
    rng: &mut OxidizeRng,
    workspace: &mut SolverWorkspace<'_>,
) -> Result<CorrelatedBernoulliSummary, ProbError> {
    let dim = thresholds_z.len();
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
    if chol_col_major.len() < dim * dim {
        return Err(ProbError::DegenerateMatrix {
            reason: "column-major Cholesky factor shorter than dim * dim",
        });
    }
    if thresholds_z.iter().any(|z| z.is_nan()) {
        return Err(ProbError::NonFiniteInput {
            context: "thresholds_z",
        });
    }

    let (eps, _) = workspace.split_floats(dim)?;
    let mut both_hits: usize = 0;
    let mut single_hits: usize = 0;
    for _ in 0..n_pairs {
        fill_standard_normals(rng, eps);
        let (pos_ok, neg_ok) = eval_col_major_pair(dim, chol_col_major, thresholds_z, eps);
        if pos_ok && neg_ok {
            both_hits += 1;
        } else if pos_ok || neg_ok {
            single_hits += 1;
        }
    }

    let m = n_pairs as f64;
    let sum_y = (both_hits as f64) + 0.5 * (single_hits as f64);
    let joint_prob = sum_y / m;
    let joint_std_error = if n_pairs > 1 {
        let sum_y_sq = (both_hits as f64) + 0.25 * (single_hits as f64);
        let sample_var = ((sum_y_sq - m * joint_prob * joint_prob) / (m - 1.0)).max(0.0);
        libm::sqrt(sample_var / m)
    } else {
        0.0
    };
    Ok(CorrelatedBernoulliSummary {
        joint_prob,
        joint_std_error,
        joint_hits: 2 * both_hits + single_hits,
        n_pairs,
        total_paths: 2 * n_pairs,
    })
}
