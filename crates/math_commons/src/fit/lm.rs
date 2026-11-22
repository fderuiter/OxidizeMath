//! Stack-allocated damped Levenberg-Marquardt nonlinear least-squares solver (`k <= 4`).
// [cite:math_commons]

use super::poly::solve_dense_small;
use crate::error::FitError;
use crate::series::PointSeries;

/// Maximum outer Levenberg-Marquardt iterations before returning [`FitError::NonConvergence`].
pub const MAX_LM_ITERATIONS: usize = 80;

const MAX_DAMPING_SUBSTEPS: usize = 12;

/// Minimizes $\sum_i (y_i - \hat{f}(x_i; \theta))^2$ over finite points in `points` on the stack.
#[verified_engine::verified]
pub fn levenberg_marquardt<P, F>(
    points: &P,
    initial_params: &[f64],
    eval_jac: F,
) -> Result<[f64; 4], FitError>
where
    P: PointSeries + ?Sized,
    F: Fn(f64, &[f64; 4]) -> (f64, [f64; 4]),
{
    levenberg_marquardt_filtered(points, initial_params, finite_xy, eval_jac)
}

/// Minimizes nonlinear least squares over observations satisfying `domain_ok(x, y)`.
#[verified_engine::verified]
pub fn levenberg_marquardt_filtered<P, F>(
    points: &P,
    initial_params: &[f64],
    domain_ok: fn(f64, f64) -> bool,
    eval_jac: F,
) -> Result<[f64; 4], FitError>
where
    P: PointSeries + ?Sized,
    F: Fn(f64, &[f64; 4]) -> (f64, [f64; 4]),
{
    let k = initial_params.len();
    if k == 0 || k > 4 {
        return Err(FitError::UnsupportedDegree { degree: k });
    }
    let valid_n = count_domain_points(points, domain_ok);
    if valid_n < k {
        return Err(FitError::InsufficientPoints {
            required: k,
            actual: valid_n,
        });
    }
    let mut theta = [0.0_f64; 4];
    theta[..k].copy_from_slice(initial_params);

    let mut current_sse = sum_squared_errors(points, domain_ok, &theta, &eval_jac);
    if !current_sse.is_finite() {
        return Err(FitError::InvalidDomain {
            reason: "non-finite initial residual sum of squares",
        });
    }
    if current_sse <= 1e-24 {
        return Ok(theta);
    }

    let mut lambda = 1e-3_f64;
    for _ in 0..MAX_LM_ITERATIONS {
        let (jtj, jtr, scaled_grad) =
            accumulate_normal_system(points, domain_ok, &theta, k, current_sse, &eval_jac)?;
        if scaled_grad <= 1e-6 || current_sse <= 1e-24 {
            return Ok(theta);
        }
        let prev_sse = current_sse;
        let (accepted, step_norm, param_norm) = try_lm_step(
            points,
            domain_ok,
            &mut theta,
            &mut current_sse,
            &mut lambda,
            k,
            &jtj,
            &jtr,
            &eval_jac,
        );
        if !accepted {
            if scaled_grad <= 1e-4 {
                return Ok(theta);
            }
            return Err(FitError::NonConvergence {
                iterations: MAX_LM_ITERATIONS,
            });
        }
        let small_step = step_norm <= 1e-8 * (1.0 + param_norm);
        let small_improve = (prev_sse - current_sse).abs() <= 1e-12 * (1.0 + current_sse);
        if lambda <= 1e4 && (small_step || small_improve) {
            return Ok(theta);
        }
    }

    let (_, _, final_grad) =
        accumulate_normal_system(points, domain_ok, &theta, k, current_sse, &eval_jac)?;
    if final_grad <= 1e-4 || current_sse <= 1e-24 {
        Ok(theta)
    } else {
        Err(FitError::NonConvergence {
            iterations: MAX_LM_ITERATIONS,
        })
    }
}

#[inline]
#[verified_engine::verified]
fn finite_xy(x: f64, y: f64) -> bool {
    x.is_finite() && y.is_finite()
}

#[verified_engine::verified]
fn count_domain_points<P: PointSeries + ?Sized>(
    points: &P,
    domain_ok: fn(f64, f64) -> bool,
) -> usize {
    let mut n = 0usize;
    for i in 0..points.len() {
        let p = points.point_at(i);
        if domain_ok(p[0], p[1]) {
            n += 1;
        }
    }
    n
}

#[allow(clippy::too_many_arguments)]
#[verified_engine::verified]
fn try_lm_step<P, F>(
    points: &P,
    domain_ok: fn(f64, f64) -> bool,
    theta: &mut [f64; 4],
    current_sse: &mut f64,
    lambda: &mut f64,
    k: usize,
    jtj: &[[f64; 4]; 4],
    jtr: &[f64; 4],
    eval_jac: &F,
) -> (bool, f64, f64)
where
    P: PointSeries + ?Sized,
    F: Fn(f64, &[f64; 4]) -> (f64, [f64; 4]),
{
    let mut step_norm = 0.0_f64;
    let mut param_norm = 0.0_f64;
    for _ in 0..MAX_DAMPING_SUBSTEPS {
        let (mut mat, rhs) = build_damped_system(jtj, jtr, k, *lambda);
        let Ok(delta) = solve_dense_small(&mut mat, &rhs, k) else {
            *lambda *= 10.0;
            continue;
        };
        let mut candidate = *theta;
        step_norm = 0.0;
        param_norm = 0.0;
        for c in 0..k {
            candidate[c] += delta[c];
            step_norm += delta[c] * delta[c];
            param_norm += theta[c] * theta[c];
        }
        step_norm = step_norm.sqrt();
        param_norm = param_norm.sqrt();

        let new_sse = sum_squared_errors(points, domain_ok, &candidate, eval_jac);
        if new_sse.is_finite() && new_sse < *current_sse {
            *theta = candidate;
            *current_sse = new_sse;
            *lambda = (*lambda * 0.25).max(1e-12);
            return (true, step_norm, param_norm);
        }
        *lambda *= 10.0;
        if *lambda > 1e12 || !lambda.is_finite() {
            break;
        }
    }
    (false, step_norm, param_norm)
}

#[verified_engine::verified]
fn build_damped_system(
    jtj: &[[f64; 4]; 4],
    jtr: &[f64; 4],
    k: usize,
    lambda: f64,
) -> ([[f64; 5]; 5], [f64; 5]) {
    let mut mat = [[0.0_f64; 5]; 5];
    let mut rhs = [0.0_f64; 5];
    for r in 0..k {
        rhs[r] = jtr[r];
        for c in 0..k {
            mat[r][c] = jtj[r][c];
        }
        let diag_scale = jtj[r][r].abs().max(1e-6);
        mat[r][r] += lambda * diag_scale;
    }
    (mat, rhs)
}

type NormalSystem = ([[f64; 4]; 4], [f64; 4], f64);

#[verified_engine::verified]
fn accumulate_normal_system<P, F>(
    points: &P,
    domain_ok: fn(f64, f64) -> bool,
    theta: &[f64; 4],
    k: usize,
    current_sse: f64,
    eval_jac: &F,
) -> Result<NormalSystem, FitError>
where
    P: PointSeries + ?Sized,
    F: Fn(f64, &[f64; 4]) -> (f64, [f64; 4]),
{
    let mut jtj = [[0.0_f64; 4]; 4];
    let mut jtr = [0.0_f64; 4];

    for idx in 0..points.len() {
        let p = points.point_at(idx);
        if !domain_ok(p[0], p[1]) {
            continue;
        }
        let (f_hat, jac) = eval_jac(p[0], theta);
        let r = p[1] - f_hat;
        if !r.is_finite() || jac[..k].iter().any(|v| !v.is_finite()) {
            return Err(FitError::SingularMatrix);
        }
        for row in 0..k {
            jtr[row] += jac[row] * r;
            for col in 0..k {
                jtj[row][col] += jac[row] * jac[col];
            }
        }
    }

    let r_norm = current_sse.sqrt().max(1e-15);
    let mut scaled_grad = 0.0_f64;
    for row in 0..k {
        let col_norm = jtj[row][row].sqrt();
        let denom = (col_norm * r_norm).max(1.0);
        scaled_grad = scaled_grad.max(jtr[row].abs() / denom);
    }
    Ok((jtj, jtr, scaled_grad))
}

#[verified_engine::verified]
fn sum_squared_errors<P, F>(
    points: &P,
    domain_ok: fn(f64, f64) -> bool,
    theta: &[f64; 4],
    eval_jac: &F,
) -> f64
where
    P: PointSeries + ?Sized,
    F: Fn(f64, &[f64; 4]) -> (f64, [f64; 4]),
{
    let mut sse = 0.0_f64;
    for idx in 0..points.len() {
        let p = points.point_at(idx);
        if !domain_ok(p[0], p[1]) {
            continue;
        }
        let (f_hat, _) = eval_jac(p[0], theta);
        if !f_hat.is_finite() {
            return f64::NAN;
        }
        let r = p[1] - f_hat;
        sse += r * r;
        if !sse.is_finite() {
            return f64::NAN;
        }
    }
    sse
}
