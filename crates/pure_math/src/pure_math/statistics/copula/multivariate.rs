//! Multivariate Gaussian copula evaluation via exact boundary reduction and Genz (1992)
//! separation-of-variables over a deterministic Richtmyer quasi-Monte Carlo sequence.

use super::bivariate::bivariate_copula_cdf;
use super::core::{Correlation, CorrelationMatrix, Probability};
use super::transforms::{fast_norm_cdf, fast_norm_inv};
use crate::error::CopulaError;
use math_commons::error::MathError;
use math_commons::math_kernel::types::{Dimension, flatten_2d_index};
use std::f64::consts::TAU;

/// Number of antithetic Richtmyer quasi-Monte Carlo pairs (`8,192` total integrand evaluations).
const RICHTMYER_PAIRS: usize = 4096;

/// Pivot threshold for semidefinite Cholesky factorization during Genz variable reordering.
const GENZ_PIVOT_EPS: f64 = 1e-12;

/// First 32 prime numbers used to construct Richtmyer irrational generators `sqrt(p_j) mod 1`.
const FIRST_PRIMES: [u32; 32] = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
    101, 103, 107, 109, 113, 127, 131,
];

fn is_prime(candidate: u32) -> bool {
    if candidate < 2 {
        return false;
    }
    let mut div = 2_u32;
    while div * div <= candidate {
        if candidate.is_multiple_of(div) {
            return false;
        }
        div += 1;
    }
    true
}

fn prime_at(index: usize) -> u32 {
    if index < FIRST_PRIMES.len() {
        return FIRST_PRIMES[index];
    }
    let mut count = FIRST_PRIMES.len();
    let mut candidate = FIRST_PRIMES[FIRST_PRIMES.len() - 1] + 2;
    while count <= index {
        if is_prime(candidate) {
            if count == index {
                return candidate;
            }
            count += 1;
        }
        candidate += 2;
    }
    candidate
}

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
        for col in 0..step {
            let l_val = chol[flatten_2d_index(col, cand, k)];
            mean_shift += l_val * y_mean[col];
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
        for j in 0..i {
            let l_ij = chol[flatten_2d_index(j, i, k)];
            diag_used += l_ij * l_ij;
            mean_shift += l_ij * y_mean[j];
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

struct QmcProblem<'a> {
    chol: &'a [f64],
    bounds: &'a [f64],
    alphas: &'a [f64],
    k: usize,
    e0: f64,
}

fn evaluate_single_qmc_point(
    prob: &QmcProblem<'_>,
    y_buf: &mut [f64],
    step_offset: f64,
    antithetic: bool,
) -> f64 {
    let k = prob.k;
    let mut prod = prob.e0;
    let mut prev_e = prob.e0;
    for i in 1..k {
        let prev_diag = prob.chol[flatten_2d_index(i - 1, i - 1, k)];
        if prev_diag > 0.0 {
            let raw = (step_offset * prob.alphas[i - 1]).fract();
            let tent = 1.0 - (2.0 * raw - 1.0).abs();
            let u_draw = if antithetic { 1.0 - tent } else { tent };
            let scaled = (u_draw.clamp(1e-15, 1.0 - 1e-15) * prev_e).clamp(1e-16, 1.0 - 1e-16);
            y_buf[i - 1] = fast_norm_inv(scaled);
        } else {
            y_buf[i - 1] = 0.0;
        }

        let mut shift = 0.0;
        for j in 0..i {
            shift += prob.chol[flatten_2d_index(j, i, k)] * y_buf[j];
        }
        let diag = prob.chol[flatten_2d_index(i, i, k)];
        let ei = if diag > 0.0 {
            fast_norm_cdf((prob.bounds[i] - shift) / diag)
        } else if shift <= prob.bounds[i] {
            1.0
        } else {
            0.0
        };
        prev_e = ei;
        prod *= ei;
        if prod == 0.0 {
            break;
        }
    }
    prod
}

fn run_richtmyer_genz_qmc(
    chol: &[f64],
    bounds: &[f64],
    alphas: &mut [f64],
    y_buf: &mut [f64],
    k: usize,
) -> f64 {
    for (j, alpha_slot) in alphas.iter_mut().take(k - 1).enumerate() {
        *alpha_slot = f64::from(prime_at(j)).sqrt().fract();
    }
    let l00 = chol[flatten_2d_index(0, 0, k)];
    let e0 = if l00 > 0.0 {
        fast_norm_cdf(bounds[0] / l00)
    } else if bounds[0] >= 0.0 {
        1.0
    } else {
        0.0
    };
    if e0 == 0.0 {
        return 0.0;
    }

    let prob = QmcProblem {
        chol,
        bounds,
        alphas,
        k,
        e0,
    };
    let mut total = 0.0;
    for n in 1..=RICHTMYER_PAIRS {
        let offset = n as f64 - 0.5;
        total += evaluate_single_qmc_point(&prob, y_buf, offset, false);
        total += evaluate_single_qmc_point(&prob, y_buf, offset, true);
    }
    total / ((2 * RICHTMYER_PAIRS) as f64)
}

fn populate_active_subproblem(
    u: &[Probability],
    corr: &CorrelationMatrix,
    bounds: &mut [f64],
    y_buf: &mut [f64],
    chol: &mut [f64],
) -> (usize, f64, f64) {
    let mut k = 0;
    let mut sum_u = 0.0;
    let mut min_u = 1.0_f64;
    for (idx, prob) in u.iter().enumerate() {
        let val = prob.value();
        if val < 1.0 {
            bounds[k] = fast_norm_inv(val);
            y_buf[k] = idx as f64;
            sum_u += val;
            min_u = min_u.min(val);
            k += 1;
        }
    }
    let mat = corr.matrix();
    for r in 0..k {
        let orig_r = y_buf[r] as usize;
        for c in 0..k {
            let orig_c = y_buf[c] as usize;
            chol[flatten_2d_index(c, r, k)] = mat[(orig_r, orig_c)];
        }
    }
    let frechet_lo = (sum_u - (k.saturating_sub(1) as f64)).max(0.0);
    (k, frechet_lo, min_u)
}

/// Evaluates `C(u_1, ..., u_d; Sigma)` with zero heap allocations using `scratch`.
#[verified_engine::verified]
pub(crate) fn evaluate_copula_cdf_with_scratch(
    u: &[Probability],
    corr: &CorrelationMatrix,
    scratch: &mut [f64],
) -> Result<Probability, CopulaError> {
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
    if u.iter().any(|p| p.value() == 0.0) {
        return Probability::new(0.0);
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
    if active_count == 0 {
        return Probability::new(1.0);
    }
    if active_count == 1 {
        return Ok(u[active_first[0]]);
    }
    if active_count == 2 {
        let i0 = active_first[0];
        let i1 = active_first[1];
        let rho = Correlation::new(corr.matrix()[(i0, i1)].clamp(-1.0, 1.0))?;
        return bivariate_copula_cdf(u[i0], u[i1], rho);
    }

    let (bounds, rest1) = scratch.split_at_mut(d);
    let (y_mean, rest2) = rest1.split_at_mut(d);
    let (y_buf, rest3) = rest2.split_at_mut(d);
    let (alphas, chol) = rest3.split_at_mut(d);
    let (k, frechet_lo, frechet_hi) = populate_active_subproblem(u, corr, bounds, y_buf, chol);
    if !factorize_genz_pivoted(chol, bounds, y_mean, k) {
        return Probability::new(frechet_lo);
    }
    let est = run_richtmyer_genz_qmc(chol, bounds, alphas, y_buf, k);
    Probability::new(est.clamp(frechet_lo, frechet_hi))
}
