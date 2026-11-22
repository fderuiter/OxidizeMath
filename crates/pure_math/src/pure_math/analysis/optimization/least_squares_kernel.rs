//! Zero-allocation Levenberg-Marquardt numerical kernel, central finite-difference
//! Jacobians, and statistical goodness-of-fit diagnostics.

use super::{
    InPlaceCalibrationMetrics, LeastSquaresConfig, LeastSquaresObjective,
    LevenbergMarquardtWorkspace, MAX_CALIBRATION_ITERATIONS, MAX_CALIBRATION_OBSERVATIONS,
    MAX_CALIBRATION_PARAMS, OptimizationError, ParameterBounds,
};
use crate::pure_math::algebra::linear_algebra::numerical::{
    moore_penrose_pseudoinverse_symmetric_in_place, solve_symmetric_cholesky_in_place,
};

#[verified_engine::verified]
pub(super) fn run_lm_in_place<O: LeastSquaresObjective>(
    cfg: &LeastSquaresConfig,
    objective: &mut O,
    initial_params: &[f64],
    bounds: &[ParameterBounds],
    ws: &mut LevenbergMarquardtWorkspace,
) -> Result<InPlaceCalibrationMetrics, OptimizationError> {
    let n = objective.observation_count();
    let p = initial_params.len();
    validate_inputs(n, p, initial_params, bounds, ws)?;
    ws.active_obs = n;
    ws.active_params = p;
    for j in 0..p {
        ws.params[j] = bounds[j].clamp(initial_params[j]);
    }
    objective.copy_observed(&mut ws.observed[..n]);
    if !objective.eval_residuals(&ws.params[..p], &mut ws.residuals[..n])
        || !all_finite(&ws.residuals[..n])
    {
        return Err(OptimizationError::NumericalInstability);
    }

    let iterations = execute_lm_loop(cfg, objective, bounds, ws, n, p)?;
    compute_final_diagnostics(cfg, objective, bounds, ws, n, p, iterations)
}

fn validate_inputs(
    n: usize,
    p: usize,
    initial_params: &[f64],
    bounds: &[ParameterBounds],
    ws: &LevenbergMarquardtWorkspace,
) -> Result<(), OptimizationError> {
    if !(1..=MAX_CALIBRATION_PARAMS).contains(&p) {
        return Err(OptimizationError::InvalidParameterCount {
            count: p,
            max: MAX_CALIBRATION_PARAMS,
        });
    }
    let min_obs = p.max(2);
    if !(min_obs..=MAX_CALIBRATION_OBSERVATIONS).contains(&n) {
        return Err(OptimizationError::InvalidObservationCount {
            count: n,
            min: min_obs,
            max: MAX_CALIBRATION_OBSERVATIONS,
        });
    }
    if bounds.len() != p {
        return Err(OptimizationError::InvalidBounds {
            index: bounds.len(),
        });
    }
    for (idx, b) in bounds.iter().enumerate() {
        if !b.lower.is_finite() || !b.upper.is_finite() || b.lower > b.upper {
            return Err(OptimizationError::InvalidBounds { index: idx });
        }
    }
    if initial_params.iter().any(|v| !v.is_finite()) {
        return Err(OptimizationError::NumericalInstability);
    }
    if n > ws.max_obs || p > ws.max_params {
        return Err(OptimizationError::WorkspaceCapacityExceeded {
            obs: n,
            params: p,
            max_obs: ws.max_obs,
            max_params: ws.max_params,
        });
    }
    Ok(())
}

fn execute_lm_loop<O: LeastSquaresObjective>(
    cfg: &LeastSquaresConfig,
    objective: &mut O,
    bounds: &[ParameterBounds],
    ws: &mut LevenbergMarquardtWorkspace,
    n: usize,
    p: usize,
) -> Result<usize, OptimizationError> {
    let max_iters = cfg.max_iterations.clamp(1, MAX_CALIBRATION_ITERATIONS);
    let mut lambda = cfg.initial_lambda.clamp(1e-12, 1e8);
    let mut rss = sum_squares(&ws.residuals[..n]);
    let mut iter_used = 0;

    for iter in 0..max_iters {
        iter_used = iter + 1;
        if !compute_central_jacobian(cfg.fd_epsilon, objective, bounds, ws, n, p) {
            return Err(OptimizationError::NumericalInstability);
        }
        assemble_normal_system(
            &ws.jacobian,
            &ws.residuals,
            n,
            p,
            &mut ws.jtj,
            &mut ws.neg_jtr,
        );
        if max_abs(&ws.neg_jtr[..p]) <= cfg.gradient_tolerance {
            break;
        }
        let accepted = try_damped_substeps(cfg, objective, bounds, ws, n, p, &mut rss, &mut lambda);
        if !accepted {
            break;
        }
        let p_norm = l2_norm(&ws.params[..p]);
        let s_norm = l2_norm(&ws.step[..p]);
        if s_norm <= cfg.step_tolerance * (p_norm + cfg.step_tolerance) || rss <= 1e-28 {
            break;
        }
    }
    Ok(iter_used)
}

#[allow(clippy::too_many_arguments)]
fn try_damped_substeps<O: LeastSquaresObjective>(
    cfg: &LeastSquaresConfig,
    objective: &mut O,
    bounds: &[ParameterBounds],
    ws: &mut LevenbergMarquardtWorkspace,
    n: usize,
    p: usize,
    rss: &mut f64,
    lambda: &mut f64,
) -> bool {
    let up = cfg.lambda_up.max(1.5);
    let down = cfg.lambda_down.clamp(0.05, 0.9);
    for _trial in 0..12 {
        build_damped_normal(&ws.jtj, p, *lambda, &mut ws.damped_jtj);
        if !solve_damped_in_place(
            &ws.damped_jtj,
            &ws.neg_jtr,
            p,
            cfg.svd_epsilon,
            &mut ws.covariance,
            &mut ws.step,
        ) {
            *lambda = (*lambda * up).min(1e12);
            continue;
        }
        for (j, bound) in bounds.iter().enumerate().take(p) {
            let candidate = ws.params[j] + ws.step[j];
            ws.trial_params[j] = bound.clamp(candidate);
            ws.step[j] = ws.trial_params[j] - ws.params[j];
        }
        if objective.eval_residuals(&ws.trial_params[..p], &mut ws.trial_residuals[..n])
            && all_finite(&ws.trial_residuals[..n])
        {
            let trial_rss = sum_squares(&ws.trial_residuals[..n]);
            if trial_rss <= *rss {
                ws.params[..p].copy_from_slice(&ws.trial_params[..p]);
                ws.residuals[..n].copy_from_slice(&ws.trial_residuals[..n]);
                *rss = trial_rss;
                *lambda = (*lambda * down).max(1e-12);
                return true;
            }
        }
        *lambda = (*lambda * up).min(1e12);
        if *lambda >= 1e12 {
            break;
        }
    }
    false
}

/// Computes central finite-difference Jacobian columns clamped to parameter box bounds.
fn compute_central_jacobian<O: LeastSquaresObjective>(
    fd_epsilon: f64,
    objective: &mut O,
    bounds: &[ParameterBounds],
    ws: &mut LevenbergMarquardtWorkspace,
    n: usize,
    p: usize,
) -> bool {
    let eps = fd_epsilon.clamp(1e-10, 1e-2);
    ws.params_plus[..p].copy_from_slice(&ws.params[..p]);
    ws.params_minus[..p].copy_from_slice(&ws.params[..p]);

    for (j, bound) in bounds.iter().enumerate().take(p) {
        let base = ws.params[j];
        let span = bound.upper - bound.lower;
        if span <= 1e-15 {
            zero_jacobian_col(&mut ws.jacobian, n, p, j);
            continue;
        }
        let raw_delta = eps * base.abs().max(1.0);
        let delta = raw_delta.min(0.25 * span).max(1e-12);
        let p_plus = bound.clamp(base + delta);
        let p_minus = bound.clamp(base - delta);
        let h = p_plus - p_minus;
        if h <= 1e-15 {
            zero_jacobian_col(&mut ws.jacobian, n, p, j);
            continue;
        }
        if !eval_perturbed_col(objective, ws, n, p, j, base, p_plus, p_minus, h) {
            return false;
        }
    }
    true
}

#[allow(clippy::too_many_arguments)]
fn eval_perturbed_col<O: LeastSquaresObjective>(
    objective: &mut O,
    ws: &mut LevenbergMarquardtWorkspace,
    n: usize,
    p: usize,
    col: usize,
    base: f64,
    p_plus: f64,
    p_minus: f64,
    h: f64,
) -> bool {
    ws.params_plus[col] = p_plus;
    let ok_plus = objective.eval_residuals(&ws.params_plus[..p], &mut ws.residuals_plus[..n]);
    ws.params_plus[col] = base;

    ws.params_minus[col] = p_minus;
    let ok_minus = objective.eval_residuals(&ws.params_minus[..p], &mut ws.residuals_minus[..n]);
    ws.params_minus[col] = base;

    if ok_plus && ok_minus {
        let inv_h = 1.0 / h;
        for i in 0..n {
            let idx = i * p + col;
            let deriv = (ws.residuals_plus[i] - ws.residuals_minus[i]) * inv_h;
            if !deriv.is_finite() {
                return false;
            }
            ws.jacobian[idx] = deriv;
        }
        return true;
    }
    fallback_one_sided_col(ws, n, p, col, base, p_plus, p_minus, ok_plus, ok_minus)
}

#[allow(clippy::too_many_arguments)]
fn fallback_one_sided_col(
    ws: &mut LevenbergMarquardtWorkspace,
    n: usize,
    p: usize,
    col: usize,
    base: f64,
    p_plus: f64,
    p_minus: f64,
    ok_plus: bool,
    ok_minus: bool,
) -> bool {
    if ok_plus && (p_plus - base).abs() > 1e-15 {
        let inv_h = 1.0 / (p_plus - base);
        for i in 0..n {
            let idx = i * p + col;
            ws.jacobian[idx] = (ws.residuals_plus[i] - ws.residuals[i]) * inv_h;
        }
        return all_finite_col(&ws.jacobian, n, p, col);
    }
    if ok_minus && (base - p_minus).abs() > 1e-15 {
        let inv_h = 1.0 / (base - p_minus);
        for i in 0..n {
            let idx = i * p + col;
            ws.jacobian[idx] = (ws.residuals[i] - ws.residuals_minus[i]) * inv_h;
        }
        return all_finite_col(&ws.jacobian, n, p, col);
    }
    false
}

fn zero_jacobian_col(jacobian: &mut [f64], n: usize, p: usize, col: usize) {
    for i in 0..n {
        let idx = i * p + col;
        jacobian[idx] = 0.0;
    }
}

fn all_finite_col(jacobian: &[f64], n: usize, p: usize, col: usize) -> bool {
    for i in 0..n {
        let idx = i * p + col;
        if !jacobian[idx].is_finite() {
            return false;
        }
    }
    true
}

fn assemble_normal_system(
    jacobian: &[f64],
    residuals: &[f64],
    n: usize,
    p: usize,
    jtj: &mut [f64],
    neg_jtr: &mut [f64],
) {
    for val in jtj.iter_mut().take(p * p) {
        *val = 0.0;
    }
    for val in neg_jtr.iter_mut().take(p) {
        *val = 0.0;
    }
    for (i, &ri) in residuals.iter().enumerate().take(n) {
        let row = i * p;
        for (r, neg_g) in neg_jtr.iter_mut().enumerate().take(p) {
            let ir = row + r;
            let j_ir = jacobian[ir];
            *neg_g -= j_ir * ri;
            let jtj_row = r * p;
            for c in 0..p {
                let ic = row + c;
                let rc = jtj_row + c;
                jtj[rc] += j_ir * jacobian[ic];
            }
        }
    }
}

fn build_damped_normal(jtj: &[f64], p: usize, lambda: f64, damped: &mut [f64]) {
    let pp = p * p;
    damped[..pp].copy_from_slice(&jtj[..pp]);
    for j in 0..p {
        let jj = j * p + j;
        let scale = jtj[jj].abs().max(1e-15);
        damped[jj] += lambda * scale;
    }
}

fn solve_damped_in_place(
    damped_jtj: &[f64],
    neg_jtr: &[f64],
    p: usize,
    svd_eps: f64,
    pinv_scratch: &mut [f64],
    step_out: &mut [f64],
) -> bool {
    if solve_symmetric_cholesky_in_place(damped_jtj, neg_jtr, p, step_out) {
        return all_finite(&step_out[..p]);
    }
    if !moore_penrose_pseudoinverse_symmetric_in_place(damped_jtj, p, svd_eps, pinv_scratch) {
        return false;
    }
    for (i, step_i) in step_out.iter_mut().enumerate().take(p) {
        let row = i * p;
        let mut sum = 0.0;
        for (j, &rhs_j) in neg_jtr.iter().enumerate().take(p) {
            let ij = row + j;
            sum += pinv_scratch[ij] * rhs_j;
        }
        *step_i = sum;
    }
    all_finite(&step_out[..p])
}

fn compute_final_diagnostics<O: LeastSquaresObjective>(
    cfg: &LeastSquaresConfig,
    objective: &mut O,
    bounds: &[ParameterBounds],
    ws: &mut LevenbergMarquardtWorkspace,
    n: usize,
    p: usize,
    iterations: usize,
) -> Result<InPlaceCalibrationMetrics, OptimizationError> {
    if !compute_central_jacobian(cfg.fd_epsilon, objective, bounds, ws, n, p) {
        return Err(OptimizationError::NumericalInstability);
    }
    assemble_normal_system(
        &ws.jacobian,
        &ws.residuals,
        n,
        p,
        &mut ws.jtj,
        &mut ws.neg_jtr,
    );
    let rss = sum_squares(&ws.residuals[..n]);
    let n_f = n as f64;
    let rmse = (rss / n_f).sqrt();
    let tss = total_sum_squares(&ws.observed[..n]);
    let r_squared = if tss > 1e-30 {
        1.0 - (rss / tss)
    } else if rss <= 1e-30 {
        1.0
    } else {
        0.0
    };
    let sigma2_mle = (rss / n_f).max(1e-30);
    let aic = n_f * sigma2_mle.ln() + 2.0 * (p as f64);

    let dof = n.saturating_sub(p).max(1);
    let s2 = rss / (dof as f64);
    let t_crit = student_t_975(dof);
    if moore_penrose_pseudoinverse_symmetric_in_place(
        &ws.jtj,
        p,
        cfg.svd_epsilon,
        &mut ws.covariance,
    ) {
        for val in ws.covariance.iter_mut().take(p * p) {
            *val *= s2;
        }
        for j in 0..p {
            let jj = j * p + j;
            let se = ws.covariance[jj].max(0.0).sqrt();
            ws.standard_errors[j] = se;
            ws.ci_95_half_widths[j] = t_crit * se;
        }
    } else {
        ws.standard_errors[..p].fill(0.0);
        ws.ci_95_half_widths[..p].fill(0.0);
    }

    Ok(InPlaceCalibrationMetrics {
        rmse,
        r_squared,
        aic,
        iterations,
        observation_count: n,
        parameter_count: p,
    })
}

/// Accurate two-sided 95% Student-t critical value $t_{0.975, \nu}$ for $\nu \ge 1$.
#[verified_engine::verified]
fn student_t_975(dof: usize) -> f64 {
    #[rustfmt::skip]
    const TABLE: [f64; 30] = [
        12.706_204_736_4, 4.302_652_729_91, 3.182_446_305_28, 2.776_445_105_2,  2.570_581_836_61,
        2.446_911_848_79, 2.364_624_251_01, 2.306_004_135_03, 2.262_157_162_74, 2.228_138_851_96,
        2.200_985_160_08, 2.178_812_829_66, 2.160_368_656_46, 2.144_786_687_92, 2.131_449_545_56,
        2.119_905_299_22, 2.109_815_577_83, 2.100_922_040_24, 2.093_024_054_41, 2.085_963_447_27,
        2.079_613_844_73, 2.073_873_067_9,  2.068_657_610_42, 2.063_898_561_63, 2.059_538_552_75,
        2.055_529_438_64, 2.051_830_516_48, 2.048_407_141_8,  2.045_229_642_13, 2.042_272_456_3,
    ];
    if (1..=30).contains(&dof) {
        return TABLE[dof - 1];
    }
    let nu = dof.max(31) as f64;
    let z: f64 = 1.959_963_984_540_054;
    let g1 = (z * z * z + z) / 4.0;
    let g2 = (5.0 * z.powi(5) + 16.0 * z.powi(3) + 3.0 * z) / 96.0;
    z + g1 / nu + g2 / (nu * nu)
}

fn all_finite(slice: &[f64]) -> bool {
    slice.iter().all(|v| v.is_finite())
}

fn sum_squares(slice: &[f64]) -> f64 {
    slice.iter().map(|v| v * v).sum()
}

fn l2_norm(slice: &[f64]) -> f64 {
    sum_squares(slice).sqrt()
}

fn max_abs(slice: &[f64]) -> f64 {
    slice.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()))
}

fn total_sum_squares(obs: &[f64]) -> f64 {
    if obs.is_empty() {
        return 0.0;
    }
    let mean = obs.iter().sum::<f64>() / (obs.len() as f64);
    obs.iter().map(|y| (y - mean) * (y - mean)).sum()
}
