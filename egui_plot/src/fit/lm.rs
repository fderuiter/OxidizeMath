//! Pure-Rust damped Levenberg-Marquardt nonlinear least-squares solver (up to 6 parameters).

use crate::fit::{MAX_FIT_PARAMS, MAX_LM_ITERATIONS, poly::solve_dense_small};

/// Evaluates model value $f(x; \theta)$ and Jacobian $\partial f / \partial \theta_j$ into `jac[..k]`.
pub(crate) type ModelEvalJacobian = fn(f64, &[f64], &mut [f64; MAX_FIT_PARAMS]) -> Option<f64>;
/// Evaluates model value $f(x; \theta)$ for cost checks.
pub(crate) type ModelEvalValue = fn(f64, &[f64]) -> Option<f64>;
type EvalValClosure<'a> = dyn Fn(f64, &[f64]) -> Option<f64> + 'a;
type EvalJacClosure<'a> = dyn Fn(f64, &[f64], &mut [f64; MAX_FIT_PARAMS]) -> Option<f64> + 'a;

/// Refines `initial_params` ($1 \le k \le 6$) on `points` using analytical-Jacobian
/// damped Levenberg-Marquardt capped at [`MAX_LM_ITERATIONS`] iterations.
pub(crate) fn levenberg_marquardt(
    points: &[[f64; 2]],
    initial_params: &[f64],
    eval_val: ModelEvalValue,
    eval_jac: ModelEvalJacobian,
) -> Option<Vec<f64>> {
    levenberg_marquardt_with(
        points,
        initial_params,
        &|x, theta| eval_val(x, theta),
        &|x, theta, jac| eval_jac(x, theta, jac),
    )
}

/// Refines `initial_params` ($1 \le k \le 6$) on `points` for an arbitrary callback `eval_fn`
/// using central finite-difference Jacobians on stack buffers.
pub(crate) fn levenberg_marquardt_numeric(
    points: &[[f64; 2]],
    initial_params: &[f64],
    eval_fn: &dyn Fn(f64, &[f64]) -> Option<f64>,
) -> Option<Vec<f64>> {
    let jac_closure = |x: f64, theta: &[f64], jac: &mut [f64; MAX_FIT_PARAMS]| -> Option<f64> {
        let y0 = eval_fn(x, theta)?;
        let k = theta.len();
        let mut work = [0.0_f64; MAX_FIT_PARAMS];
        work[..k].copy_from_slice(theta);
        for j in 0..k {
            let h = 1e-6 * theta[j].abs().max(1.0);
            work[j] = theta[j] + h;
            let yp = eval_fn(x, &work[..k])?;
            work[j] = theta[j] - h;
            let ym = eval_fn(x, &work[..k])?;
            work[j] = theta[j];
            jac[j] = (yp - ym) / (2.0 * h);
        }
        Some(y0)
    };
    levenberg_marquardt_with(points, initial_params, eval_fn, &jac_closure)
}

fn levenberg_marquardt_with(
    points: &[[f64; 2]],
    initial_params: &[f64],
    eval_val: &EvalValClosure<'_>,
    eval_jac: &EvalJacClosure<'_>,
) -> Option<Vec<f64>> {
    let k = initial_params.len();
    if k == 0 || k > MAX_FIT_PARAMS || points.len() < k {
        return None;
    }
    let mut theta = initial_params.to_vec();
    let mut current_sse = sum_squared_errors_with(points, &theta, eval_val)?;
    let mut lambda = 1e-3_f64;
    let mut candidate = vec![0.0; k];

    for _ in 0..MAX_LM_ITERATIONS {
        if current_sse <= 1e-28 {
            break;
        }
        let Some((jtj, jtr, max_grad)) = accumulate_normal_system(points, &theta, k, eval_jac)
        else {
            break;
        };
        if max_grad <= 1e-12 {
            break;
        }
        let sys = LmStepSystem {
            jtj: &jtj,
            jtr: &jtr,
            current_sse,
        };
        let Some((accepted, next_sse, step_norm)) =
            try_lm_step(points, &theta, &mut candidate, &sys, &mut lambda, eval_val)
        else {
            break;
        };
        if accepted {
            theta.copy_from_slice(&candidate);
            current_sse = next_sse;
            let param_norm: f64 = theta.iter().map(|v| v * v).sum::<f64>().sqrt();
            if step_norm <= 1e-10 * (1.0 + param_norm) {
                break;
            }
        }
    }
    Some(theta)
}

struct LmStepSystem<'a> {
    jtj: &'a [[f64; MAX_FIT_PARAMS]; MAX_FIT_PARAMS],
    jtr: &'a [f64; MAX_FIT_PARAMS],
    current_sse: f64,
}

fn try_lm_step(
    points: &[[f64; 2]],
    theta: &[f64],
    candidate: &mut [f64],
    sys: &LmStepSystem<'_>,
    lambda: &mut f64,
    eval_val: &dyn Fn(f64, &[f64]) -> Option<f64>,
) -> Option<(bool, f64, f64)> {
    let k = theta.len();
    for _sub in 0..6 {
        let mut mat = [[0.0_f64; MAX_FIT_PARAMS]; MAX_FIT_PARAMS];
        let mut rhs = [0.0_f64; MAX_FIT_PARAMS];
        for r in 0..k {
            mat[r][..k].copy_from_slice(&sys.jtj[r][..k]);
            let diag = sys.jtj[r][r].abs().max(1e-6);
            mat[r][r] += *lambda * diag + (*lambda * 1e-8);
            rhs[r] = sys.jtr[r];
        }
        if let Some(delta) = solve_dense_small(&mut mat, &mut rhs, k) {
            let mut step_sq = 0.0;
            for j in 0..k {
                candidate[j] = theta[j] + delta[j];
                step_sq += delta[j] * delta[j];
            }
            if candidate.iter().all(|v| v.is_finite())
                && let Some(cand_sse) = sum_squared_errors_with(points, candidate, eval_val)
                && cand_sse < sys.current_sse
            {
                *lambda = (*lambda * 0.25).max(1e-11);
                return Some((true, cand_sse, step_sq.sqrt()));
            }
        }
        *lambda *= 8.0;
        if *lambda > 1e12 || !lambda.is_finite() {
            return None;
        }
    }
    Some((false, sys.current_sse, 0.0))
}

type NormalSystem = (
    [[f64; MAX_FIT_PARAMS]; MAX_FIT_PARAMS],
    [f64; MAX_FIT_PARAMS],
    f64,
);

fn accumulate_normal_system(
    points: &[[f64; 2]],
    theta: &[f64],
    k: usize,
    eval_jac: &EvalJacClosure<'_>,
) -> Option<NormalSystem> {
    let mut jtj = [[0.0_f64; MAX_FIT_PARAMS]; MAX_FIT_PARAMS];
    let mut jtr = [0.0_f64; MAX_FIT_PARAMS];
    let mut jac = [0.0_f64; MAX_FIT_PARAMS];
    for p in points {
        let y_hat = eval_jac(p[0], theta, &mut jac)?;
        let residual = p[1] - y_hat;
        if !residual.is_finite() || !jac[..k].iter().all(|v| v.is_finite()) {
            return None;
        }
        for r in 0..k {
            jtr[r] += jac[r] * residual;
            for c in r..k {
                jtj[r][c] += jac[r] * jac[c];
            }
        }
    }
    let mut max_grad = 0.0_f64;
    for r in 0..k {
        max_grad = max_grad.max(jtr[r].abs());
        let row_r = jtj[r];
        for (row_c, &val) in jtj.iter_mut().take(k).skip(r + 1).zip(&row_r[(r + 1)..k]) {
            row_c[r] = val;
        }
    }
    Some((jtj, jtr, max_grad))
}

pub(crate) fn sum_squared_errors(
    points: &[[f64; 2]],
    theta: &[f64],
    eval_val: ModelEvalValue,
) -> Option<f64> {
    sum_squared_errors_with(points, theta, &|x, p| eval_val(x, p))
}

pub(crate) fn sum_squared_errors_with(
    points: &[[f64; 2]],
    theta: &[f64],
    eval_val: &dyn Fn(f64, &[f64]) -> Option<f64>,
) -> Option<f64> {
    let mut sse = 0.0_f64;
    for p in points {
        let y_hat = eval_val(p[0], theta)?;
        let diff = p[1] - y_hat;
        let sq = diff * diff;
        if !sq.is_finite() {
            return None;
        }
        sse += sq;
    }
    sse.is_finite().then_some(sse)
}
