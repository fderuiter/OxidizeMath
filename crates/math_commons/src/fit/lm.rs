//! Stack-allocated damped Levenberg-Marquardt nonlinear least-squares solvers (`k <= 6`).
// [cite:math_commons]

use super::micro_solvers::solve_marquardt_ldlt;
use super::poly::symmetrize_upper;
use crate::error::FitError;
use crate::series::PointSeries;
use math_scalars::RealField;

/// Maximum outer Levenberg-Marquardt iterations before returning [`FitError::NonConvergence`].
pub const MAX_LM_ITERATIONS: usize = 80;

/// Maximum number of free parameters supported by stack-backed Levenberg-Marquardt regression.
pub const MAX_FIT_PARAMS: usize = 6;

const MAX_DAMPING_SUBSTEPS: usize = 12;

/// Minimizes $\sum_i (y_i - \hat{f}(x_i; \theta))^2$ via damped normal equations $(J^\top J + \lambda \operatorname{diag}(J^\top J))\delta = J^\top r$.
///
/// Reference: Levenberg (1944), DOI: 10.1090/qam/10666; Marquardt (1963), DOI: 10.1137/0111030;
/// [math_commons specification](../../../../docs/math_specs/math_commons.md).
///
/// # Stability
/// Diagonal Marquardt damping $\lambda > 0$ solved via stack symmetric $L D L^\top$ regularizes rank-deficient Jacobians up to [`MAX_LM_ITERATIONS`] outer steps.
///
/// # Domain of Validity
/// Requires `1 <= initial_params.len() <= 4`, at least `initial_params.len()` finite `(x, y)` pairs, and finite `eval_jac` outputs.
#[cfg_attr(feature = "std", verified_engine::verified)]
pub fn levenberg_marquardt<T, P, F>(
    points: &P,
    initial_params: &[T],
    eval_jac: F,
) -> Result<[T; 4], FitError>
where
    T: RealField,
    P: PointSeries<T> + ?Sized,
    F: Fn(T, &[T; 4]) -> (T, [T; 4]),
{
    lm_core::<T, 4, P, _, _>(
        points,
        initial_params,
        finite_xy::<T>,
        false,
        &|x, th| eval_jac(x, th).0,
        &eval_jac,
    )
}

/// Minimizes $\sum_{i:\,\mathrm{domain\_ok}(x_i,y_i)} (y_i - \hat{f}(x_i; \theta))^2$ via damped Levenberg-Marquardt updates.
///
/// Reference: Marquardt (1963), DOI: 10.1137/0111030; [math_commons specification](../../../../docs/math_specs/math_commons.md).
///
/// # Stability
/// Rejects steps producing non-finite residuals and increases damping $\lambda$ across up to `12` substeps per outer iteration.
///
/// # Domain of Validity
/// Requires `1 <= initial_params.len() <= 4` and at least `initial_params.len()` observations satisfying `domain_ok(x, y)`.
#[cfg_attr(feature = "std", verified_engine::verified)]
pub fn levenberg_marquardt_filtered<T, P, F>(
    points: &P,
    initial_params: &[T],
    domain_ok: fn(T, T) -> bool,
    eval_jac: F,
) -> Result<[T; 4], FitError>
where
    T: RealField,
    P: PointSeries<T> + ?Sized,
    F: Fn(T, &[T; 4]) -> (T, [T; 4]),
{
    lm_core::<T, 4, P, _, _>(
        points,
        initial_params,
        domain_ok,
        false,
        &|x, th| eval_jac(x, th).0,
        &eval_jac,
    )
}

/// Minimizes $\sum_i (y_i - \hat{f}(x_i; \theta))^2$ ($1 \le k \le 6$) using central finite-difference Jacobians $\partial_j \hat{f} \approx (\hat{f}(\theta + h_j e_j) - \hat{f}(\theta - h_j e_j)) / (2 h_j)$.
///
/// Reference: Moré (1978), DOI: 10.1007/BFb0067700; [math_commons specification](../../../../docs/math_specs/math_commons.md).
///
/// # Stability
/// Step sizes scale with parameter magnitude $h_j = \epsilon^{1/3}\max(1, |\theta_j|)$ to balance truncation and cancellation error.
///
/// # Domain of Validity
/// Requires `1 <= initial_params.len() <= 6`, at least `initial_params.len()` finite points, and smooth finite `eval_fn` values near $\theta$.
#[cfg_attr(feature = "std", verified_engine::verified)]
pub fn levenberg_marquardt_numeric<T, P, F>(
    points: &P,
    initial_params: &[T],
    eval_fn: F,
) -> Result<[T; MAX_FIT_PARAMS], FitError>
where
    T: RealField,
    P: PointSeries<T> + ?Sized,
    F: Fn(T, &[T]) -> Option<T>,
{
    let k = initial_params.len();
    let eval_val =
        |x: T, theta: &[T; MAX_FIT_PARAMS]| -> T { eval_fn(x, &theta[..k]).unwrap_or(T::NAN) };
    let eval_jac = |x: T, theta: &[T; MAX_FIT_PARAMS]| -> (T, [T; MAX_FIT_PARAMS]) {
        numeric_central_jacobian(x, theta, k, &eval_fn)
    };
    lm_core::<T, MAX_FIT_PARAMS, P, _, _>(
        points,
        initial_params,
        finite_xy::<T>,
        true,
        &eval_val,
        &eval_jac,
    )
}

/// Evaluates $\sum_i (y_i - \hat{f}(x_i; \theta))^2$ over finite points in `points`, returning
/// `None` if no finite points exist or any evaluation produces a non-finite residual.
#[cfg_attr(feature = "std", verified_engine::verified)]
pub fn sum_squared_errors_numeric<T, P, F>(points: &P, params: &[T], eval_fn: F) -> Option<T>
where
    T: RealField,
    P: PointSeries<T> + ?Sized,
    F: Fn(T, &[T]) -> Option<T>,
{
    sum_squared_errors_slice(points, params, &eval_fn)
}

fn sum_squared_errors_slice<T, P, F>(points: &P, params: &[T], eval_fn: &F) -> Option<T>
where
    T: RealField,
    P: PointSeries<T> + ?Sized,
    F: Fn(T, &[T]) -> Option<T>,
{
    let mut sse = T::ZERO;
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

fn numeric_central_jacobian<T, F>(
    x: T,
    theta: &[T; MAX_FIT_PARAMS],
    k: usize,
    eval_fn: &F,
) -> (T, [T; MAX_FIT_PARAMS])
where
    T: RealField,
    F: Fn(T, &[T]) -> Option<T>,
{
    let Some(y0) = eval_fn(x, &theta[..k]) else {
        return (T::NAN, [T::ZERO; MAX_FIT_PARAMS]);
    };
    if !y0.is_finite() {
        return (T::NAN, [T::ZERO; MAX_FIT_PARAMS]);
    }
    let mut work = *theta;
    let mut jac = [T::ZERO; MAX_FIT_PARAMS];
    let step_scale = T::EPSILON.cbrt().max(T::from_f64(1e-6));
    for j in 0..k {
        let h = step_scale * theta[j].abs().max(T::ONE);
        work[j] = theta[j] + h;
        let Some(yp) = eval_fn(x, &work[..k]) else {
            return (T::NAN, [T::ZERO; MAX_FIT_PARAMS]);
        };
        work[j] = theta[j] - h;
        let Some(ym) = eval_fn(x, &work[..k]) else {
            return (T::NAN, [T::ZERO; MAX_FIT_PARAMS]);
        };
        work[j] = theta[j];
        jac[j] = (yp - ym) / (T::TWO * h);
    }
    (y0, jac)
}

fn lm_core<T: RealField, const M: usize, P, V, J>(
    points: &P,
    initial_params: &[T],
    domain_ok: fn(T, T) -> bool,
    allow_improved: bool,
    eval_val: &V,
    eval_jac: &J,
) -> Result<[T; M], FitError>
where
    P: PointSeries<T> + ?Sized,
    V: Fn(T, &[T; M]) -> T,
    J: Fn(T, &[T; M]) -> (T, [T; M]),
{
    let (k, mut theta, initial_sse) = init_lm_state(points, initial_params, domain_ok, eval_val)?;
    let sse_floor = (T::EPSILON * T::EPSILON).max(T::from_f64(1e-24));
    if initial_sse <= sse_floor {
        return Ok(theta);
    }

    let mut current_sse = initial_sse;
    let mut lambda = T::from_f64(1e-3);
    let grad_tol = T::from_f64(1e-6).max(T::EPSILON.sqrt());
    let step_tol = T::from_f64(1e-8).max(T::EPSILON.sqrt());
    let imp_tol = T::from_f64(1e-12).max(T::EPSILON * T::from_f64(16.0));

    for _ in 0..MAX_LM_ITERATIONS {
        let (jtj, jtr, scaled_grad) =
            accumulate_normal_system(points, domain_ok, &theta, k, current_sse, eval_jac)?;
        if scaled_grad <= grad_tol || current_sse <= sse_floor {
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
        let small_step = step_norm <= step_tol * (T::ONE + param_norm);
        let small_improve = (prev_sse - current_sse).abs() <= imp_tol * (T::ONE + current_sse);
        if lambda <= T::from_f64(1e4) && (small_step || small_improve) {
            return Ok(theta);
        }
    }

    let (_, _, final_grad) =
        accumulate_normal_system(points, domain_ok, &theta, k, current_sse, eval_jac)?;
    finish_lm(theta, final_grad, current_sse, initial_sse, allow_improved)
}

fn finish_lm<T: RealField, const M: usize>(
    theta: [T; M],
    grad: T,
    current_sse: T,
    initial_sse: T,
    allow_improved: bool,
) -> Result<[T; M], FitError> {
    let sse_floor = (T::EPSILON * T::EPSILON).max(T::from_f64(1e-24));
    if grad <= T::from_f64(1e-4)
        || current_sse <= sse_floor
        || (allow_improved && current_sse < initial_sse)
    {
        Ok(theta)
    } else {
        Err(FitError::NonConvergence {
            iterations: MAX_LM_ITERATIONS,
        })
    }
}

fn init_lm_state<T: RealField, const M: usize, P, V>(
    points: &P,
    initial_params: &[T],
    domain_ok: fn(T, T) -> bool,
    eval_val: &V,
) -> Result<(usize, [T; M], T), FitError>
where
    P: PointSeries<T> + ?Sized,
    V: Fn(T, &[T; M]) -> T,
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
    let mut theta = [T::ZERO; M];
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
fn finite_xy<T: RealField>(x: T, y: T) -> bool {
    x.is_finite() && y.is_finite()
}

fn count_domain_points<T: RealField, P: PointSeries<T> + ?Sized>(
    points: &P,
    domain_ok: fn(T, T) -> bool,
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

struct LmLinearSystem<'a, T, const M: usize> {
    k: usize,
    jtj: &'a [[T; M]; M],
    jtr: &'a [T; M],
}

fn try_lm_step<T: RealField, const M: usize, P, V>(
    points: &P,
    domain_ok: fn(T, T) -> bool,
    theta: &mut [T; M],
    current_sse: &mut T,
    lambda: &mut T,
    sys: &LmLinearSystem<'_, T, M>,
    eval_val: &V,
) -> (bool, T, T)
where
    P: PointSeries<T> + ?Sized,
    V: Fn(T, &[T; M]) -> T,
{
    let k = sys.k;
    let mut step_norm = T::ZERO;
    let mut param_norm = T::ZERO;
    let ten = T::from_f64(10.0);
    let quarter = T::from_f64(0.25);
    let diag_floor = T::from_f64(1e-6);
    for _ in 0..MAX_DAMPING_SUBSTEPS {
        let Ok(delta) = solve_marquardt_ldlt(sys.jtj, sys.jtr, k, *lambda, diag_floor) else {
            *lambda *= ten;
            continue;
        };
        let mut candidate = *theta;
        step_norm = T::ZERO;
        param_norm = T::ZERO;
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
            *lambda = (*lambda * quarter).max(T::from_f64(1e-12));
            return (true, step_norm, param_norm);
        }
        *lambda *= ten;
        if *lambda > T::from_f64(1e12) || !lambda.is_finite() {
            break;
        }
    }
    (false, step_norm, param_norm)
}

type NormalSystem<T, const M: usize> = ([[T; M]; M], [T; M], T);

fn accumulate_normal_system<T: RealField, const M: usize, P, J>(
    points: &P,
    domain_ok: fn(T, T) -> bool,
    theta: &[T; M],
    k: usize,
    current_sse: T,
    eval_jac: &J,
) -> Result<NormalSystem<T, M>, FitError>
where
    P: PointSeries<T> + ?Sized,
    J: Fn(T, &[T; M]) -> (T, [T; M]),
{
    let mut jtj = [[T::ZERO; M]; M];
    let mut jtr = [T::ZERO; M];

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
    symmetrize_upper(&mut jtj, k);

    let r_norm = current_sse.sqrt().max(T::from_f64(1e-15));
    let mut scaled_grad = T::ZERO;
    for row in 0..k {
        let col_norm = jtj[row][row].sqrt();
        let denom = (col_norm * r_norm).max(T::ONE);
        scaled_grad = scaled_grad.max(jtr[row].abs() / denom);
    }
    Ok((jtj, jtr, scaled_grad))
}

fn sum_squared_errors<T: RealField, const M: usize, P, V>(
    points: &P,
    domain_ok: fn(T, T) -> bool,
    theta: &[T; M],
    eval_val: &V,
) -> T
where
    P: PointSeries<T> + ?Sized,
    V: Fn(T, &[T; M]) -> T,
{
    let mut sse = T::ZERO;
    for idx in 0..points.len() {
        let p = points.point_at(idx);
        if !domain_ok(p[0], p[1]) {
            continue;
        }
        let f_hat = eval_val(p[0], theta);
        if !f_hat.is_finite() {
            return T::NAN;
        }
        let r = p[1] - f_hat;
        sse += r * r;
        if !sse.is_finite() {
            return T::NAN;
        }
    }
    sse
}
