//! Row-major positive-semidefinite Genz (1992) Richtmyer cubature with shift error bounds.

use crate::cubature::CubatureEstimate;
use crate::error::ProbError;
use crate::math::{
    flatten_row_major, standard_normal_cdf, standard_normal_inv_cdf, Z_95_TWO_SIDED,
};
use crate::workspace::SolverWorkspace;
use oxidize_core::rng::{baker_tent, richtmyer_generator, OxidizeRng};

struct SemidefProblem<'a> {
    chol_row_major: &'a [f64],
    bounds: &'a [f64],
    alphas: &'a [f64],
    shift: Option<&'a [f64]>,
    dim: usize,
    e0: f64,
}

fn eval_semidef_point(
    prob: &SemidefProblem<'_>,
    y_buf: &mut [f64],
    step_offset: f64,
    antithetic: bool,
) -> f64 {
    let k = prob.dim;
    let mut prod = prob.e0;
    let mut prev_e = prob.e0;
    for i in 1..k {
        let prev_diag = prob.chol_row_major[flatten_row_major(i - 1, i - 1, k)];
        if prev_diag > 0.0 {
            let base = step_offset * prob.alphas[i - 1] + prob.shift.map_or(0.0, |s| s[i - 1]);
            let raw = base - libm::floor(base);
            let tent = baker_tent(raw);
            let u_draw = if antithetic { 1.0 - tent } else { tent };
            let scaled = (u_draw.clamp(1e-15, 1.0 - 1e-15) * prev_e).clamp(1e-16, 1.0 - 1e-16);
            y_buf[i - 1] = standard_normal_inv_cdf(scaled);
        } else {
            y_buf[i - 1] = 0.0;
        }

        let mut mean_shift = 0.0;
        for (j, &y_val) in y_buf.iter().enumerate().take(i) {
            mean_shift += prob.chol_row_major[flatten_row_major(i, j, k)] * y_val;
        }
        let diag = prob.chol_row_major[flatten_row_major(i, i, k)];
        let ei = if diag > 0.0 {
            standard_normal_cdf((prob.bounds[i] - mean_shift) / diag)
        } else if mean_shift <= prob.bounds[i] {
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

fn run_semidef_shift_pass(prob: &SemidefProblem<'_>, n_pairs: usize, y_buf: &mut [f64]) -> f64 {
    let mut total = 0.0;
    for n in 1..=n_pairs {
        let offset = (n as f64) - 0.5;
        total += eval_semidef_point(prob, y_buf, offset, false);
        total += eval_semidef_point(prob, y_buf, offset, true);
    }
    (total / ((2 * n_pairs) as f64)).clamp(0.0, 1.0)
}

#[inline]
const fn const_estimate(
    val: f64,
    n_points: usize,
    n_shifts: usize,
    n_evals: usize,
) -> CubatureEstimate {
    CubatureEstimate {
        value: val,
        std_error: 0.0,
        lower_95: val,
        upper_95: val,
        n_points,
        n_shifts,
        n_evaluations: n_evals,
    }
}

fn validate_semidef_inputs(
    chol_row_major: &[f64],
    bounds: &[f64],
    dim: usize,
    n_pairs: usize,
) -> Result<(), ProbError> {
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
    if chol_row_major.len() < dim * dim || bounds.len() < dim {
        return Err(ProbError::DegenerateMatrix {
            reason: "semidefinite Cholesky or bounds shorter than dim",
        });
    }
    if bounds[..dim].iter().any(|b| b.is_nan()) {
        return Err(ProbError::NonFiniteInput {
            context: "semidefinite Genz bounds",
        });
    }
    Ok(())
}

fn summarize_shifted_semidef(
    base_prob: &SemidefProblem<'_>,
    n_pairs: usize,
    n_shifts: usize,
    seed: u64,
    y_buf: &mut [f64],
    shift_buf: &mut [f64],
) -> CubatureEstimate {
    let mut rng = OxidizeRng::from_seed(seed);
    let (mut mean, mut m2) = (0.0, 0.0);
    for s_idx in 0..n_shifts {
        for slot in shift_buf.iter_mut() {
            *slot = rng.next_f64_open01();
        }
        let prob = SemidefProblem {
            shift: Some(shift_buf),
            ..*base_prob
        };
        let est = run_semidef_shift_pass(&prob, n_pairs, y_buf);
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
    let pts = 2 * n_pairs;
    CubatureEstimate {
        value: val,
        std_error: se,
        lower_95: (val - half).clamp(0.0, 1.0),
        upper_95: (val + half).clamp(0.0, 1.0),
        n_points: pts,
        n_shifts,
        n_evaluations: pts.saturating_mul(n_shifts),
    }
}

/// Evaluates $\Phi_d((-\infty, b]; L L^\top)$ from a row-major (`row * dim + col`)
/// positive-semidefinite Cholesky factor `chol_row_major` using `n_pairs` antithetic half-offset
/// Baker-tent Richtmyer pairs and optional Cranley-Patterson `n_shifts` shift error bounds.
///
/// Requires `2 * dim` floats in [`SolverWorkspace`] when `n_shifts == 0` (`3 * dim` when `n_shifts >= 1`).
pub fn genz_semidefinite_richtmyer_qmc(
    chol_row_major: &[f64],
    bounds: &[f64],
    dim: usize,
    n_pairs: usize,
    n_shifts: usize,
    seed: u64,
    workspace: &mut SolverWorkspace<'_>,
) -> Result<CubatureEstimate, ProbError> {
    validate_semidef_inputs(chol_row_major, bounds, dim, n_pairs)?;
    let l00 = chol_row_major[0];
    let e0 = if l00 > 0.0 {
        standard_normal_cdf(bounds[0] / l00)
    } else if bounds[0] >= 0.0 {
        1.0
    } else {
        0.0
    };
    let pts = 2 * n_pairs;
    if dim == 1 || e0 == 0.0 {
        return Ok(const_estimate(
            e0,
            pts,
            n_shifts.max(1),
            usize::from(dim == 1),
        ));
    }

    let qmc_dim = dim - 1;
    let (alphas, mut rem1) = workspace.split_floats(qmc_dim)?;
    let (y_buf, mut rem2) = rem1.split_floats(dim)?;
    for (j, slot) in alphas.iter_mut().enumerate() {
        *slot = richtmyer_generator(j);
    }
    let base_prob = SemidefProblem {
        chol_row_major,
        bounds,
        alphas,
        shift: None,
        dim,
        e0,
    };
    if n_shifts == 0 {
        let val = run_semidef_shift_pass(&base_prob, n_pairs, y_buf);
        return Ok(const_estimate(val, pts, 1, pts));
    }
    let (shift_buf, _) = rem2.split_floats(qmc_dim)?;
    Ok(summarize_shifted_semidef(
        &base_prob, n_pairs, n_shifts, seed, y_buf, shift_buf,
    ))
}
