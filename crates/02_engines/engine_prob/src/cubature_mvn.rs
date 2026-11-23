//! Genz (1992) separation-of-variables Richtmyer QMC for Gaussian copula and MVN CDFs.

use crate::cubature::CubatureEstimate;
use crate::error::ProbError;
use crate::math::{standard_normal_cdf, standard_normal_inv_cdf, Z_95_TWO_SIDED};
use crate::workspace::SolverWorkspace;
use oxidize_core::rng::{OxidizeRng, RichtmyerSequence, MAX_QMC_DIM};

const GENZ_EPS: f64 = 1e-15;

/// Minimum `f64` scratch slots required in [`SolverWorkspace`] for `dim`-dimensional MVN/copula CDF.
#[inline]
#[must_use]
pub const fn workspace_floats_for_mvn_cdf(dim: usize) -> usize {
    4 * dim
}

fn genz_integrand_point(
    dim: usize,
    chol_lower: &[f64],
    upper_z: &[f64],
    u_point: &[f64],
    y_work: &mut [f64],
) -> f64 {
    let l00 = chol_lower[0];
    let e0 = if upper_z[0].is_infinite() && upper_z[0].is_sign_positive() {
        1.0
    } else {
        standard_normal_cdf(upper_z[0] / l00)
    };
    let mut prod = e0;
    for i in 1..dim {
        if prod < 1e-16 {
            return 0.0;
        }
        let u_prev = u_point[i - 1].clamp(0.0, 1.0);
        let p_prev =
            (u_prev * e0_at(i - 1, chol_lower, upper_z, y_work)).clamp(GENZ_EPS, 1.0 - GENZ_EPS);
        y_work[i - 1] = standard_normal_inv_cdf(p_prev);

        let e_i = e0_at(i, chol_lower, upper_z, y_work);
        prod *= e_i;
    }
    let _ = e0;
    prod
}

#[inline]
fn e0_at(row: usize, chol_lower: &[f64], upper_z: &[f64], y_work: &[f64]) -> f64 {
    let dim = upper_z.len();
    let z_i = upper_z[row];
    if z_i.is_infinite() {
        return if z_i.is_sign_positive() { 1.0 } else { 0.0 };
    }
    let mut mean_shift = 0.0;
    let row_off = row * dim;
    for j in 0..row {
        mean_shift += chol_lower[row_off + j] * y_work[j];
    }
    let diag = chol_lower[row_off + row];
    standard_normal_cdf((z_i - mean_shift) / diag)
}

fn run_single_genz_shift(
    dim: usize,
    chol_lower: &[f64],
    upper_z: &[f64],
    n_points: usize,
    shift: Option<&[f64]>,
    u_buf: &mut [f64],
    y_buf: &mut [f64],
) -> Result<f64, ProbError> {
    let qmc_dim = (dim - 1).max(1);
    let mut seq =
        RichtmyerSequence::with_baker_tent(qmc_dim, true).ok_or(ProbError::InvalidDimension {
            dim,
            max_dim: MAX_QMC_DIM,
        })?;
    let mut sum = 0.0;
    for _ in 0..n_points {
        match shift {
            Some(s) => seq.next_shifted_into(&s[..qmc_dim], &mut u_buf[..qmc_dim]),
            None => seq.next_into(&mut u_buf[..qmc_dim]),
        }
        sum += genz_integrand_point(dim, chol_lower, upper_z, u_buf, y_buf);
    }
    Ok((sum / (n_points as f64)).clamp(0.0, 1.0))
}

fn validate_mvn_inputs(
    chol_lower: &[f64],
    upper_z: &[f64],
    n_points: usize,
) -> Result<bool, ProbError> {
    let dim = upper_z.len();
    if dim == 0 || dim > MAX_QMC_DIM {
        return Err(ProbError::InvalidDimension {
            dim,
            max_dim: MAX_QMC_DIM,
        });
    }
    if chol_lower.len() < dim * dim {
        return Err(ProbError::DegenerateMatrix {
            reason: "Cholesky factor slice shorter than dim * dim",
        });
    }
    if n_points == 0 {
        return Err(ProbError::InvalidCount {
            name: "n_points",
            value: 0,
        });
    }
    let mut has_neg_inf = false;
    for (i, &z_i) in upper_z.iter().enumerate() {
        if z_i.is_nan() {
            return Err(ProbError::NonFiniteInput {
                context: "MVN upper integration bound",
            });
        }
        let diag = chol_lower[crate::math::flatten_row_major(i, i, dim)];
        if !diag.is_finite() || diag <= 0.0 {
            return Err(ProbError::DegenerateMatrix {
                reason: "Cholesky diagonal must be positive and finite",
            });
        }
        if z_i.is_infinite() && z_i.is_sign_negative() {
            has_neg_inf = true;
        }
    }
    Ok(has_neg_inf)
}

#[inline]
const fn exact_estimate(
    value: f64,
    n_points: usize,
    n_shifts: usize,
    n_evals: usize,
) -> CubatureEstimate {
    CubatureEstimate {
        value,
        std_error: 0.0,
        lower_95: value,
        upper_95: value,
        n_points,
        n_shifts,
        n_evaluations: n_evals,
    }
}

/// Evaluates the multivariate normal orthant probability $\Phi_d((-\infty, z]; L L^\top)$ via
/// Genz separation-of-variables and Baker-tent Richtmyer cubature over [`SolverWorkspace`].
///
/// Setting `n_shifts = 0` runs a single unshifted pass; `n_shifts >= 2` reports Cranley-Patterson
/// shift standard errors and 95% confidence bounds clamped to $[0, 1]$.
pub fn mvn_genz_richtmyer_cdf(
    chol_lower: &[f64],
    upper_z: &[f64],
    n_points: usize,
    n_shifts: usize,
    seed: u64,
    workspace: &mut SolverWorkspace<'_>,
) -> Result<CubatureEstimate, ProbError> {
    if validate_mvn_inputs(chol_lower, upper_z, n_points)? {
        return Ok(exact_estimate(0.0, n_points, n_shifts.max(1), 0));
    }
    let dim = upper_z.len();
    if dim == 1 {
        return Ok(exact_estimate(
            standard_normal_cdf(upper_z[0] / chol_lower[0]),
            1,
            1,
            1,
        ));
    }

    let (u_buf, mut rem1) = workspace.split_floats(dim)?;
    let (y_buf, mut rem2) = rem1.split_floats(dim)?;
    let (shift_buf, _) = rem2.split_floats(dim)?;
    if n_shifts == 0 {
        let val = run_single_genz_shift(dim, chol_lower, upper_z, n_points, None, u_buf, y_buf)?;
        return Ok(exact_estimate(val, n_points, 1, n_points));
    }

    let mut rng = OxidizeRng::from_seed(seed);
    let (mut mean, mut m2) = (0.0, 0.0);
    for s_idx in 0..n_shifts {
        for slot in shift_buf.iter_mut().take(dim) {
            *slot = rng.next_f64_open01();
        }
        let est = run_single_genz_shift(
            dim,
            chol_lower,
            upper_z,
            n_points,
            Some(shift_buf),
            u_buf,
            y_buf,
        )?;
        let delta = est - mean;
        mean += delta / ((s_idx + 1) as f64);
        m2 += delta * (est - mean);
    }
    let se = if n_shifts > 1 {
        libm::sqrt((m2 / ((n_shifts - 1) as f64)) / (n_shifts as f64))
    } else {
        0.0
    };
    let val = mean.clamp(0.0, 1.0);
    let half = Z_95_TWO_SIDED * se;
    Ok(CubatureEstimate {
        value: val,
        std_error: se,
        lower_95: (val - half).clamp(0.0, 1.0),
        upper_95: (val + half).clamp(0.0, 1.0),
        n_points,
        n_shifts,
        n_evaluations: n_points.saturating_mul(n_shifts),
    })
}

/// Evaluates the $d$-dimensional Gaussian copula CDF $C_\Sigma(u_1, \dots, u_d)$ from Cholesky
/// factor `chol_lower` using Baker-tent Richtmyer cubature and optional shift error bounds.
pub fn gaussian_copula_richtmyer_cdf(
    chol_lower: &[f64],
    u: &[f64],
    n_points: usize,
    n_shifts: usize,
    seed: u64,
    workspace: &mut SolverWorkspace<'_>,
) -> Result<CubatureEstimate, ProbError> {
    let dim = u.len();
    if dim == 0 || dim > MAX_QMC_DIM {
        return Err(ProbError::InvalidDimension {
            dim,
            max_dim: MAX_QMC_DIM,
        });
    }
    for &ui in u {
        if ui.is_nan() {
            return Err(ProbError::NonFiniteInput {
                context: "copula marginal u",
            });
        }
        if ui <= 0.0 {
            return Ok(CubatureEstimate {
                value: 0.0,
                std_error: 0.0,
                lower_95: 0.0,
                upper_95: 0.0,
                n_points,
                n_shifts: n_shifts.max(1),
                n_evaluations: 0,
            });
        }
    }
    if u.iter().all(|&ui| ui >= 1.0) {
        return Ok(CubatureEstimate {
            value: 1.0,
            std_error: 0.0,
            lower_95: 1.0,
            upper_95: 1.0,
            n_points,
            n_shifts: n_shifts.max(1),
            n_evaluations: 0,
        });
    }
    let (z_buf, mut sub_ws) = workspace.split_floats(dim)?;
    for (j, &ui) in u.iter().enumerate() {
        z_buf[j] = if ui >= 1.0 {
            f64::INFINITY
        } else {
            standard_normal_inv_cdf(ui.clamp(GENZ_EPS, 1.0 - GENZ_EPS))
        };
    }
    mvn_genz_richtmyer_cdf(chol_lower, z_buf, n_points, n_shifts, seed, &mut sub_ws)
}
