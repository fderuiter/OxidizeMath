//! Pure-Rust damped Levenberg-Marquardt nonlinear least-squares solver.

use crate::fit::{MAX_LM_ITERATIONS, poly::solve_dense_small};

/// Evaluates model value $f(x; \theta)$ and Jacobian $\partial f / \partial \theta_j$ into `jac[..k]`.
pub(crate) type ModelEvalJacobian = fn(f64, &[f64], &mut [f64; 4]) -> Option<f64>;
/// Evaluates model value $f(x; \theta)$ for cost checks.
pub(crate) type ModelEvalValue = fn(f64, &[f64]) -> Option<f64>;

/// Refines `params` ($1 \le k \le 4$) on `points` using damped Levenberg-Marquardt
/// capped at [`MAX_LM_ITERATIONS`] iterations.
pub(crate) fn levenberg_marquardt(
    points: &[[f64; 2]],
    initial_params: &[f64],
    eval_val: ModelEvalValue,
    eval_jac: ModelEvalJacobian,
) -> Option<Vec<f64>> {
    let k = initial_params.len();
    if k == 0 || k > 4 || points.len() < k {
        return None;
    }
    let mut theta = initial_params.to_vec();
    let mut current_sse = sum_squared_errors(points, &theta, eval_val)?;
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
        if let Some((accepted, next_sse, step_norm)) = try_lm_step(
            points,
            &theta,
            &mut candidate,
            &LmStepSystem {
                jtj: &jtj,
                jtr: &jtr,
                current_sse,
            },
            &mut lambda,
            eval_val,
        ) {
            if accepted {
                theta.copy_from_slice(&candidate);
                current_sse = next_sse;
                let param_norm: f64 = theta.iter().map(|v| v * v).sum::<f64>().sqrt();
                if step_norm <= 1e-10 * (1.0 + param_norm) {
                    break;
                }
            }
        } else {
            break;
        }
    }
    Some(theta)
}

struct LmStepSystem<'a> {
    jtj: &'a [[f64; 4]; 4],
    jtr: &'a [f64; 4],
    current_sse: f64,
}

fn try_lm_step(
    points: &[[f64; 2]],
    theta: &[f64],
    candidate: &mut [f64],
    sys: &LmStepSystem<'_>,
    lambda: &mut f64,
    eval_val: ModelEvalValue,
) -> Option<(bool, f64, f64)> {
    let k = theta.len();
    for _sub in 0..6 {
        let mut mat = [[0.0_f64; 5]; 5];
        let mut rhs = [0.0_f64; 5];
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
                && let Some(cand_sse) = sum_squared_errors(points, candidate, eval_val)
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

fn accumulate_normal_system(
    points: &[[f64; 2]],
    theta: &[f64],
    k: usize,
    eval_jac: ModelEvalJacobian,
) -> Option<([[f64; 4]; 4], [f64; 4], f64)> {
    let mut jtj = [[0.0_f64; 4]; 4];
    let mut jtr = [0.0_f64; 4];
    let mut jac = [0.0_f64; 4];
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
