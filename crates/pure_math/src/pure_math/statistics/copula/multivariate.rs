//! Multivariate Gaussian copula evaluation via exact boundary reduction and Genz (1992)
//! separation-of-variables delegated to [`engine_prob::cubature::genz_semidefinite_richtmyer_qmc`].

use super::bivariate::bivariate_copula_cdf;
use super::core::{Correlation, CorrelationMatrix, Probability};
use super::transforms::{fast_norm_cdf, fast_norm_inv};
use crate::error::CopulaError;
use engine_prob::cubature::{CubatureEstimate, genz_semidefinite_richtmyer_qmc};
use engine_prob::workspace::SolverWorkspace;
use math_commons::error::MathError;
use math_commons::math_kernel::types::{Dimension, flatten_2d_index};
use oxidize_core::rng::BASE_SEED;
use std::f64::consts::TAU;

/// Number of antithetic Richtmyer quasi-Monte Carlo pairs (`8,192` total integrand evaluations).
const RICHTMYER_PAIRS: usize = 4096;

/// Pivot threshold for semidefinite Cholesky factorization during Genz variable reordering.
const GENZ_PIVOT_EPS: f64 = 1e-12;

/// Scratch length in `f64` elements required by [`evaluate_copula_cdf_with_scratch`] for `d` legs.
#[verified_engine::verified]
pub(crate) const fn required_scratch_len(d: usize) -> usize {
    d * d + 4 * d
}

fn swap_pivoted_leg(chol: &mut [f64], bounds: &mut [f64], k: usize, i: usize, best: usize) {
    if i == best {
        return;
    }
    bounds.swap(i, best);
    for col in 0..k {
        chol.swap(flatten_2d_index(col, i, k), flatten_2d_index(col, best, k));
    }
    for row in 0..k {
        chol.swap(flatten_2d_index(i, row, k), flatten_2d_index(best, row, k));
    }
}

fn select_genz_pivot(chol: &[f64], bounds: &[f64], y_mean: &[f64], k: usize, step: usize) -> usize {
    let mut best_idx = step;
    let mut best_prob = 2.0;
    let mut best_var = -1.0;

    for cand in step..k {
        let mut mean_shift = 0.0;
        let mut used_var = 0.0;
        for (col, &ym) in y_mean.iter().enumerate().take(step) {
            let l_val = chol[flatten_2d_index(col, cand, k)];
            mean_shift += l_val * ym;
            used_var += l_val * l_val;
        }
        let rem_var = chol[flatten_2d_index(cand, cand, k)] - used_var;
        let prob = if rem_var > GENZ_PIVOT_EPS {
            fast_norm_cdf((bounds[cand] - mean_shift) / rem_var.sqrt())
        } else if mean_shift <= bounds[cand] {
            1.0
        } else {
            0.0
        };
        if prob < best_prob - 1e-14 || ((prob - best_prob).abs() <= 1e-14 && rem_var > best_var) {
            best_prob = prob;
            best_idx = cand;
            best_var = rem_var;
        }
    }
    best_idx
}

fn factorize_genz_pivoted(
    chol: &mut [f64],
    bounds: &mut [f64],
    y_mean: &mut [f64],
    k: usize,
) -> bool {
    let inv_sqrt_2pi = 1.0 / TAU.sqrt();
    for i in 0..k {
        let best = select_genz_pivot(chol, bounds, y_mean, k, i);
        swap_pivoted_leg(chol, bounds, k, i, best);

        let mut diag_used = 0.0;
        let mut mean_shift = 0.0;
        for (j, &ym) in y_mean.iter().enumerate().take(i) {
            let l_ij = chol[flatten_2d_index(j, i, k)];
            diag_used += l_ij * l_ij;
            mean_shift += l_ij * ym;
        }
        let rem = chol[flatten_2d_index(i, i, k)] - diag_used;
        if rem > GENZ_PIVOT_EPS {
            let l_ii = rem.sqrt();
            chol[flatten_2d_index(i, i, k)] = l_ii;
            for r in (i + 1)..k {
                let mut dot = 0.0;
                for j in 0..i {
                    dot += chol[flatten_2d_index(j, r, k)] * chol[flatten_2d_index(j, i, k)];
                }
                let idx = flatten_2d_index(i, r, k);
                chol[idx] = (chol[idx] - dot) / l_ii;
            }
            let z_hat = (bounds[i] - mean_shift) / l_ii;
            let p_i = fast_norm_cdf(z_hat);
            if p_i <= 0.0 {
                return false;
            }
            y_mean[i] = -inv_sqrt_2pi * (-0.5 * z_hat * z_hat).exp() / p_i;
        } else {
            for r in i..k {
                chol[flatten_2d_index(i, r, k)] = 0.0;
            }
            y_mean[i] = 0.0;
        }
    }
    true
}

fn populate_active_subproblem(
    u: &[Probability],
    corr: &CorrelationMatrix,
    bounds: &mut [f64],
    idx_scratch: &mut [f64],
    chol: &mut [f64],
) -> (usize, f64, f64) {
    let mut k = 0;
    let mut sum_u = 0.0;
    let mut min_u = 1.0_f64;
    for (idx, prob) in u.iter().enumerate() {
        let val = prob.value();
        if val < 1.0 {
            bounds[k] = fast_norm_inv(val);
            idx_scratch[k] = idx as f64;
            sum_u += val;
            min_u = min_u.min(val);
            k += 1;
        }
    }
    let mat = corr.matrix();
    for r in 0..k {
        let orig_r = idx_scratch[r] as usize;
        for c in 0..k {
            let orig_c = idx_scratch[c] as usize;
            chol[flatten_2d_index(c, r, k)] = mat[(orig_r, orig_c)];
        }
    }
    let frechet_lo = (sum_u - (k.saturating_sub(1) as f64)).max(0.0);
    (k, frechet_lo, min_u)
}

#[inline]
const fn exact_cubature_estimate(val: f64, n_shifts: usize) -> CubatureEstimate {
    CubatureEstimate {
        value: val,
        std_error: 0.0,
        lower_95: val,
        upper_95: val,
        n_points: 1,
        n_shifts: if n_shifts == 0 { 1 } else { n_shifts },
        n_evaluations: 1,
    }
}

fn check_boundary_reduction(
    u: &[Probability],
    corr: &CorrelationMatrix,
    n_shifts: usize,
) -> Result<Option<(Probability, CubatureEstimate)>, CopulaError> {
    if u.iter().any(|p| p.value() == 0.0) {
        let p = Probability::new(0.0)?;
        return Ok(Some((p, exact_cubature_estimate(0.0, n_shifts))));
    }
    let mut active_first = [0_usize; 2];
    let mut active_count = 0_usize;
    for (idx, p) in u.iter().enumerate() {
        if p.value() < 1.0 {
            if active_count < 2 {
                active_first[active_count] = idx;
            }
            active_count += 1;
        }
    }
    match active_count {
        0 => {
            let p = Probability::new(1.0)?;
            Ok(Some((p, exact_cubature_estimate(1.0, n_shifts))))
        }
        1 => {
            let p = u[active_first[0]];
            Ok(Some((p, exact_cubature_estimate(p.value(), n_shifts))))
        }
        2 => {
            let (i0, i1) = (active_first[0], active_first[1]);
            let rho = Correlation::new(corr.matrix()[(i0, i1)].clamp(-1.0, 1.0))?;
            let p = bivariate_copula_cdf(u[i0], u[i1], rho)?;
            Ok(Some((p, exact_cubature_estimate(p.value(), n_shifts))))
        }
        _ => Ok(None),
    }
}

/// Evaluates `C(u_1, ..., u_d; Sigma)` and optional Cranley-Patterson shift bounds via [`engine_prob`].
#[verified_engine::verified]
pub(crate) fn evaluate_copula_cdf_estimate_with_scratch(
    u: &[Probability],
    corr: &CorrelationMatrix,
    n_shifts: usize,
    seed: u64,
    scratch: &mut [f64],
) -> Result<(Probability, CubatureEstimate), CopulaError> {
    let d = corr.dimension();
    if u.len() != d {
        return Err(CopulaError::Math(MathError::DimensionMismatch {
            expected: Dimension(d),
            actual: Dimension(u.len()),
        }));
    }
    let needed = required_scratch_len(d);
    if scratch.len() < needed {
        return Err(CopulaError::Math(MathError::DimensionMismatch {
            expected: Dimension(needed),
            actual: Dimension(scratch.len()),
        }));
    }
    if let Some(reduced) = check_boundary_reduction(u, corr, n_shifts)? {
        return Ok(reduced);
    }

    let (bounds, rest1) = scratch.split_at_mut(d);
    let (work_3d, chol) = rest1.split_at_mut(3 * d);
    let (k, frechet_lo, frechet_hi) =
        populate_active_subproblem(u, corr, bounds, &mut work_3d[..d], chol);
    if !factorize_genz_pivoted(chol, bounds, &mut work_3d[..d], k) {
        let p = Probability::new(frechet_lo)?;
        return Ok((p, exact_cubature_estimate(frechet_lo, n_shifts)));
    }

    let mut ws = SolverWorkspace::from_floats(work_3d);
    let raw_est = genz_semidefinite_richtmyer_qmc(
        &chol[..k * k],
        &bounds[..k],
        k,
        RICHTMYER_PAIRS,
        n_shifts,
        seed,
        &mut ws,
    )
    .map_err(|e| CopulaError::Math(MathError::from(e.to_string())))?;

    let clamped = raw_est.value.clamp(frechet_lo, frechet_hi);
    let prob = Probability::new(clamped)?;
    let est = CubatureEstimate {
        value: clamped,
        lower_95: raw_est.lower_95.clamp(frechet_lo, frechet_hi),
        upper_95: raw_est.upper_95.clamp(frechet_lo, frechet_hi),
        ..raw_est
    };
    Ok((prob, est))
}

/// Evaluates `C(u_1, ..., u_d; Sigma)` with zero heap allocations using `scratch`.
#[verified_engine::verified]
pub(crate) fn evaluate_copula_cdf_with_scratch(
    u: &[Probability],
    corr: &CorrelationMatrix,
    scratch: &mut [f64],
) -> Result<Probability, CopulaError> {
    evaluate_copula_cdf_estimate_with_scratch(u, corr, 0, BASE_SEED, scratch).map(|(p, _)| p)
}
