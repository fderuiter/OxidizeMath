//! Stack-allocated damped Levenberg-Marquardt nonlinear least-squares solvers (`k <= 6`).
// [cite:math_commons]

use super::poly::solve_dense_n_raw;
use crate::error::FitError;
use crate::series::PointSeries;

/// Maximum outer Levenberg-Marquardt iterations before returning [`FitError::NonConvergence`].
pub const MAX_LM_ITERATIONS: usize = 80;

/// Maximum number of free parameters supported by stack-backed Levenberg-Marquardt regression.
pub const MAX_FIT_PARAMS: usize = 6;

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
    lm_core::<4, P, _, _>(
        points,
        initial_params,
        finite_xy,
        false,
        &|x, th| eval_jac(x, th).0,
        &eval_jac,
    )
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
    lm_core::<4, P, _, _>(
        points,
        initial_params,
        domain_ok,
        false,
        &|x, th| eval_jac(x, th).0,
        &eval_jac,
    )
}

/// Minimizes nonlinear least squares ($1 \le k \le 6$) for an arbitrary scalar callback `eval_fn`
/// using central finite-difference Jacobians on stack buffers with zero heap allocations.
#[verified_engine::verified]
pub fn levenberg_marquardt_numeric<P, F>(
    points: &P,
    initial_params: &[f64],
    eval_fn: F,
) -> Result<[f64; MAX_FIT_PARAMS], FitError>
where
    P: PointSeries + ?Sized,
    F: Fn(f64, &[f64]) -> Option<f64>,
{
    let k = initial_params.len();
    let eval_val = |x: f64, theta: &[f64; MAX_FIT_PARAMS]| -> f64 {
        eval_fn(x, &theta[..k]).unwrap_or(f64::NAN)
    };
    let eval_jac = |x: f64, theta: &[f64; MAX_FIT_PARAMS]| -> (f64, [f64; MAX_FIT_PARAMS]) {
        numeric_central_jacobian(x, theta, k, &eval_fn)
    };
    lm_core::<MAX_FIT_PARAMS, P, _, _>(
        points,
        initial_params,
        finite_xy,
        true,
        &eval_val,
        &eval_jac,
    )
}

/// Evaluates $\sum_i (y_i - \hat{f}(x_i; \theta))^2$ over finite points in `points`, returning
/// `None` if no finite points exist or any evaluation produces a non-finite residual.
#[verified_engine::verified]
pub fn sum_squared_errors_numeric<P, F>(points: &P, params: &[f64], eval_fn: F) -> Option<f64>
where
    P: PointSeries + ?Sized,
    F: Fn(f64, &[f64]) -> Option<f64>,
{
    sum_squared_errors_slice(points, params, &eval_fn)
}

fn sum_squared_errors_slice<P, F>(points: &P, params: &[f64], eval_fn: &F) -> Option<f64>
where
    P: PointSeries + ?Sized,
    F: Fn(f64, &[f64]) -> Option<f64>,
{
    let mut sse = 0.0_f64;
    let mut any = false;
    for idx in 0..points.len() {
        let p = points.point_at(idx);
        if !finite_xy(p[0], p[1]) {
            continue;
        }
        let y_hat = eval_fn(p[0], params)?;
        if !y_hat.is_finite() {
            return None;
        }
        let r = p[1] - y_hat;
        sse += r * r;
        if !sse.is_finite() {
            return None;
        }
        any = true;
    }
    any.then_some(sse)
}

fn numeric_central_jacobian<F>(
    x: f64,
    theta: &[f64; MAX_FIT_PARAMS],
    k: usize,
    eval_fn: &F,
) -> (f64, [f64; MAX_FIT_PARAMS])
where
    F: Fn(f64, &[f64]) -> Option<f64>,
{
    let Some(y0) = eval_fn(x, &theta[..k]) else {
        return (f64::NAN, [0.0; MAX_FIT_PARAMS]);
    };
    if !y0.is_finite() {
        return (f64::NAN, [0.0; MAX_FIT_PARAMS]);
    }
    let mut work = *theta;
    let mut jac = [0.0_f64; MAX_FIT_PARAMS];
    for j in 0..k {
        let h = 1e-6 * theta[j].abs().max(1.0);
        work[j] = theta[j] + h;
        let Some(yp) = eval_fn(x, &work[..k]) else {
            return (f64::NAN, [0.0; MAX_FIT_PARAMS]);
        };
        work[j] = theta[j] - h;
        let Some(ym) = eval_fn(x, &work[..k]) else {
            return (f64::NAN, [0.0; MAX_FIT_PARAMS]);
        };
        work[j] = theta[j];
        jac[j] = (yp - ym) / (2.0 * h);
    }
    (y0, jac)
}

fn lm_core<const M: usize, P, V, J>(
    points: &P,
    initial_params: &[f64],
    domain_ok: fn(f64, f64) -> bool,
    allow_improved: bool,
    eval_val: &V,
    eval_jac: &J,
) -> Result<[f64; M], FitError>
where
    P: PointSeries + ?Sized,
    V: Fn(f64, &[f64; M]) -> f64,
    J: Fn(f64, &[f64; M]) -> (f64, [f64; M]),
{
    let (k, mut theta, initial_sse) = init_lm_state(points, initial_params, domain_ok, eval_val)?;
    if initial_sse <= 1e-24 {
        return Ok(theta);
    }

    let mut current_sse = initial_sse;
    let mut lambda = 1e-3_f64;
    for _ in 0..MAX_LM_ITERATIONS {
        let (jtj, jtr, scaled_grad) =
            accumulate_normal_system(points, domain_ok, &theta, k, current_sse, eval_jac)?;
        if scaled_grad <= 1e-6 || current_sse <= 1e-24 {
            return Ok(theta);
        }
        let prev_sse = current_sse;
        let sys = LmLinearSystem {
            k,
            jtj: &jtj,
            jtr: &jtr,
        };
        let (accepted, step_norm, param_norm) = try_lm_step(
            points,
            domain_ok,
            &mut theta,
            &mut current_sse,
            &mut lambda,
            &sys,
            eval_val,
        );
        if !accepted {
            return finish_lm(theta, scaled_grad, current_sse, initial_sse, allow_improved);
        }
        let small_step = step_norm <= 1e-8 * (1.0 + param_norm);
        let small_improve = (prev_sse - current_sse).abs() <= 1e-12 * (1.0 + current_sse);
        if lambda <= 1e4 && (small_step || small_improve) {
            return Ok(theta);
        }
    }

    let (_, _, final_grad) =
        accumulate_normal_system(points, domain_ok, &theta, k, current_sse, eval_jac)?;
    finish_lm(theta, final_grad, current_sse, initial_sse, allow_improved)
}

fn finish_lm<const M: usize>(
    theta: [f64; M],
    grad: f64,
    current_sse: f64,
    initial_sse: f64,
    allow_improved: bool,
) -> Result<[f64; M], FitError> {
    if grad <= 1e-4 || current_sse <= 1e-24 || (allow_improved && current_sse < initial_sse) {
        Ok(theta)
    } else {
        Err(FitError::NonConvergence {
            iterations: MAX_LM_ITERATIONS,
        })
    }
}

fn init_lm_state<const M: usize, P, V>(
    points: &P,
    initial_params: &[f64],
    domain_ok: fn(f64, f64) -> bool,
    eval_val: &V,
) -> Result<(usize, [f64; M], f64), FitError>
where
    P: PointSeries + ?Sized,
    V: Fn(f64, &[f64; M]) -> f64,
{
    let k = initial_params.len();
    if k == 0 || k > M {
        return Err(FitError::UnsupportedDegree { degree: k });
    }
    let valid_n = count_domain_points(points, domain_ok);
    if valid_n < k {
        return Err(FitError::InsufficientPoints {
            required: k,
            actual: valid_n,
        });
    }
    let mut theta = [0.0_f64; M];
    theta[..k].copy_from_slice(initial_params);
    let current_sse = sum_squared_errors(points, domain_ok, &theta, eval_val);
    if !current_sse.is_finite() {
        return Err(FitError::InvalidDomain {
            reason: "non-finite initial residual sum of squares",
        });
    }
    Ok((k, theta, current_sse))
}

#[inline]
fn finite_xy(x: f64, y: f64) -> bool {
    x.is_finite() && y.is_finite()
}

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

struct LmLinearSystem<'a, const M: usize> {
    k: usize,
    jtj: &'a [[f64; M]; M],
    jtr: &'a [f64; M],
}

fn try_lm_step<const M: usize, P, V>(
    points: &P,
    domain_ok: fn(f64, f64) -> bool,
    theta: &mut [f64; M],
    current_sse: &mut f64,
    lambda: &mut f64,
    sys: &LmLinearSystem<'_, M>,
    eval_val: &V,
) -> (bool, f64, f64)
where
    P: PointSeries + ?Sized,
    V: Fn(f64, &[f64; M]) -> f64,
{
    let k = sys.k;
    let mut step_norm = 0.0_f64;
    let mut param_norm = 0.0_f64;
    for _ in 0..MAX_DAMPING_SUBSTEPS {
        let (mut mat, rhs) = build_damped_system(sys.jtj, sys.jtr, k, *lambda);
        let Ok(delta) = solve_dense_n_raw(&mut mat, &rhs, k) else {
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

        let new_sse = sum_squared_errors(points, domain_ok, &candidate, eval_val);
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

fn build_damped_system<const M: usize>(
    jtj: &[[f64; M]; M],
    jtr: &[f64; M],
    k: usize,
    lambda: f64,
) -> ([[f64; M]; M], [f64; M]) {
    let mut mat = [[0.0_f64; M]; M];
    let mut rhs = [0.0_f64; M];
    for r in 0..k {
        rhs[r] = jtr[r];
        mat[r][..k].copy_from_slice(&jtj[r][..k]);
        let diag_scale = jtj[r][r].abs().max(1e-6);
        mat[r][r] += lambda * diag_scale;
    }
    (mat, rhs)
}

type NormalSystem<const M: usize> = ([[f64; M]; M], [f64; M], f64);

fn accumulate_normal_system<const M: usize, P, J>(
    points: &P,
    domain_ok: fn(f64, f64) -> bool,
    theta: &[f64; M],
    k: usize,
    current_sse: f64,
    eval_jac: &J,
) -> Result<NormalSystem<M>, FitError>
where
    P: PointSeries + ?Sized,
    J: Fn(f64, &[f64; M]) -> (f64, [f64; M]),
{
    let mut jtj = [[0.0_f64; M]; M];
    let mut jtr = [0.0_f64; M];

    for idx in 0..points.len() {
        let p = points.point_at(idx);
        if !domain_ok(p[0], p[1]) {
            continue;
        }
        let (f_hat, jac) = eval_jac(p[0], theta);
        let r = p[1] - f_hat;
        if !r.is_finite() {
            return Err(FitError::SingularMatrix);
        }
        for row in 0..k {
            let jr = jac[row];
            if !jr.is_finite() {
                return Err(FitError::SingularMatrix);
            }
            jtr[row] += jr * r;
            for col in row..k {
                jtj[row][col] += jr * jac[col];
            }
        }
    }
    super::mirror_upper_triangle(&mut jtj, k);

    let r_norm = current_sse.sqrt().max(1e-15);
    let mut scaled_grad = 0.0_f64;
    for row in 0..k {
        let col_norm = jtj[row][row].sqrt();
        let denom = (col_norm * r_norm).max(1.0);
        scaled_grad = scaled_grad.max(jtr[row].abs() / denom);
    }
    Ok((jtj, jtr, scaled_grad))
}

fn sum_squared_errors<const M: usize, P, V>(
    points: &P,
    domain_ok: fn(f64, f64) -> bool,
    theta: &[f64; M],
    eval_val: &V,
) -> f64
where
    P: PointSeries + ?Sized,
    V: Fn(f64, &[f64; M]) -> f64,
{
    let mut sse = 0.0_f64;
    for idx in 0..points.len() {
        let p = points.point_at(idx);
        if !domain_ok(p[0], p[1]) {
            continue;
        }
        let f_hat = eval_val(p[0], theta);
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
