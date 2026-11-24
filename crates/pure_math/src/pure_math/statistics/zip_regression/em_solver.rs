//! Expectation-Maximization (EM) covariate solver for Zero-Inflated Poisson regression.

use super::core::Count;
use super::link_functions::{LogLink, LogitLink};
use super::regression::{ZipRegression, simple_zip_fit};
use crate::error::ZipError;
use nalgebra::{DMatrix, DVector};

/// Fitted covariate parameters and diagnostics returned by [`ZipRegression::fit`].
#[derive(Debug, Clone, PartialEq)]
pub struct ZipFitResult {
    /// Fitted count-model (log-link) regression coefficients $\hat{\beta}$.
    pub beta: DVector<f64>,
    /// Fitted zero-inflation (logit-link) regression coefficients $\hat{\alpha}$.
    pub alpha: DVector<f64>,
    /// Converged observed-data log-likelihood $\ell(\hat{\beta}, \hat{\alpha})$.
    pub log_likelihood: f64,
    /// Number of EM iterations executed prior to convergence.
    pub iterations: usize,
    /// Akaike Information Criterion $-2\ell + 2(p + q)$.
    pub aic: f64,
    /// Bayesian Information Criterion $-2\ell + \ln(n)(p + q)$.
    pub bic: f64,
}

pub(crate) fn fit_zip_em(
    model: &ZipRegression,
    counts: &[Count],
    x_mat: &DMatrix<f64>,
    z_mat: &DMatrix<f64>,
    max_iter: usize,
    tol: f64,
) -> Result<ZipFitResult, ZipError> {
    validate_em_inputs(x_mat, z_mat, max_iter, tol)?;
    let (mut beta, mut alpha) = initialize_coefficients(counts, x_mat.ncols(), z_mat.ncols())?;
    let mut prev_ll = model.log_likelihood(&beta, &alpha);
    if !prev_ll.is_finite() {
        return Err(ZipError::ConvergenceFailed {
            iterations: 0,
            reason: "initial log-likelihood is non-finite".to_string(),
        });
    }

    let n = counts.len();
    let mut responsibilities = vec![0.0_f64; n];

    for iter in 1..=max_iter {
        e_step(counts, x_mat, z_mat, &beta, &alpha, &mut responsibilities);
        let next_alpha = m_step_logit(z_mat, &responsibilities, &alpha)?;
        let next_beta = m_step_poisson(x_mat, counts, &responsibilities, &beta)?;
        let ll_new = model.log_likelihood(&next_beta, &next_alpha);
        if !ll_new.is_finite() {
            return Err(ZipError::ConvergenceFailed {
                iterations: iter,
                reason: "non-finite log-likelihood encountered during EM".to_string(),
            });
        }
        let param_delta = (&next_beta - &beta).norm() + (&next_alpha - &alpha).norm();
        let ll_delta = (ll_new - prev_ll).abs();
        beta = next_beta;
        alpha = next_alpha;
        prev_ll = ll_new;

        if ll_delta <= tol * (1.0 + prev_ll.abs()) && param_delta <= tol.sqrt() {
            let k_params = (beta.len() + alpha.len()) as f64;
            let nf = n as f64;
            return Ok(ZipFitResult {
                beta,
                alpha,
                log_likelihood: ll_new,
                iterations: iter,
                aic: -2.0 * ll_new + 2.0 * k_params,
                bic: -2.0 * ll_new + nf.ln() * k_params,
            });
        }
    }

    Err(ZipError::ConvergenceFailed {
        iterations: max_iter,
        reason: format!("EM did not converge within {max_iter} iterations"),
    })
}

fn validate_em_inputs(
    x_mat: &DMatrix<f64>,
    z_mat: &DMatrix<f64>,
    max_iter: usize,
    tol: f64,
) -> Result<(), ZipError> {
    if max_iter == 0 || !tol.is_finite() || tol <= 0.0 {
        return Err(ZipError::ConvergenceFailed {
            iterations: max_iter,
            reason: "max_iter must be >= 1 and tolerance must be positive".to_string(),
        });
    }
    if x_mat.ncols() == 0 || z_mat.ncols() == 0 {
        return Err(ZipError::InvalidDimensions {
            expected: "at least 1 predictor column in X and Z".to_string(),
            actual: format!("X cols={}, Z cols={}", x_mat.ncols(), z_mat.ncols()),
        });
    }
    if x_mat.iter().any(|v| !v.is_finite()) || z_mat.iter().any(|v| !v.is_finite()) {
        return Err(ZipError::ConvergenceFailed {
            iterations: 0,
            reason: "design matrix contains non-finite entries".to_string(),
        });
    }
    Ok(())
}

fn initialize_coefficients(
    counts: &[Count],
    p_cols: usize,
    q_cols: usize,
) -> Result<(DVector<f64>, DVector<f64>), ZipError> {
    let init = simple_zip_fit(counts)?;
    let mut beta = DVector::zeros(p_cols);
    let mut alpha = DVector::zeros(q_cols);
    beta[0] = init.lambda.value().max(0.05).ln();
    let rho_clamped = init.rho.value().clamp(0.02, 0.98);
    alpha[0] = (rho_clamped / (1.0 - rho_clamped)).ln();
    Ok((beta, alpha))
}

fn e_step(
    counts: &[Count],
    x_mat: &DMatrix<f64>,
    z_mat: &DMatrix<f64>,
    beta: &DVector<f64>,
    alpha: &DVector<f64>,
    out_w: &mut [f64],
) {
    let beta_t = beta.transpose();
    let alpha_t = alpha.transpose();
    for (i, &c) in counts.iter().enumerate() {
        if c.value() > 0 {
            out_w[i] = 0.0;
        } else {
            let eta = x_mat.row(i).dot(&beta_t).clamp(-20.0, 20.0);
            let gamma = z_mat.row(i).dot(&alpha_t).clamp(-20.0, 20.0);
            let lambda = LogLink::link(eta);
            let rho = LogitLink::link(gamma);
            let denom = rho + (1.0 - rho) * (-lambda).exp();
            out_w[i] = if denom > 0.0 {
                (rho / denom).clamp(0.0, 1.0)
            } else {
                0.0
            };
        }
    }
}

fn logit_objective(z_mat: &DMatrix<f64>, w: &[f64], alpha: &DVector<f64>) -> f64 {
    let alpha_t = alpha.transpose();
    let mut obj = 0.0;
    for (i, &wi) in w.iter().enumerate() {
        let g = z_mat.row(i).dot(&alpha_t).clamp(-30.0, 30.0);
        let softplus = if g > 0.0 {
            g + (-g).exp().ln_1p()
        } else {
            g.exp().ln_1p()
        };
        obj += wi * g - softplus;
    }
    obj - 1e-6 * alpha.norm_squared()
}

fn m_step_logit(
    z_mat: &DMatrix<f64>,
    w: &[f64],
    alpha_init: &DVector<f64>,
) -> Result<DVector<f64>, ZipError> {
    let q = alpha_init.len();
    let mut alpha = alpha_init.clone();
    for _ in 0..8 {
        let mut grad = -2e-6 * &alpha;
        let mut hess = DMatrix::identity(q, q) * 1e-5;
        for (i, &wi) in w.iter().enumerate() {
            let row = z_mat.row(i).transpose();
            let rho = LogitLink::link(row.dot(&alpha)).clamp(1e-8, 1.0 - 1e-8);
            grad += &row * (wi - rho);
            hess += (&row * row.transpose()) * (rho * (1.0 - rho));
        }
        if grad.norm() < 1e-9 {
            break;
        }
        let Some(step) = hess.lu().solve(&grad) else {
            break;
        };
        let base_obj = logit_objective(z_mat, w, &alpha);
        let mut scale = 1.0;
        while scale >= 1e-4 {
            let cand = &alpha + scale * &step;
            if logit_objective(z_mat, w, &cand) >= base_obj - 1e-12 {
                alpha = cand;
                break;
            }
            scale *= 0.5;
        }
    }
    Ok(alpha)
}

fn poisson_objective(
    x_mat: &DMatrix<f64>,
    counts: &[Count],
    w: &[f64],
    beta: &DVector<f64>,
) -> f64 {
    let beta_t = beta.transpose();
    let mut obj = 0.0;
    for (i, &c) in counts.iter().enumerate() {
        let active_w = (1.0 - w[i]).max(0.0);
        let eta = x_mat.row(i).dot(&beta_t).clamp(-20.0, 20.0);
        obj += active_w * (c.as_f64() * eta - eta.exp());
    }
    obj - 1e-6 * beta.norm_squared()
}

fn m_step_poisson(
    x_mat: &DMatrix<f64>,
    counts: &[Count],
    w: &[f64],
    beta_init: &DVector<f64>,
) -> Result<DVector<f64>, ZipError> {
    let p = beta_init.len();
    let mut beta = beta_init.clone();
    for _ in 0..8 {
        let mut grad = -2e-6 * &beta;
        let mut hess = DMatrix::identity(p, p) * 1e-5;
        for (i, &c) in counts.iter().enumerate() {
            let active_w = (1.0 - w[i]).max(0.0);
            if active_w <= 0.0 {
                continue;
            }
            let row = x_mat.row(i).transpose();
            let eta = row.dot(&beta).clamp(-20.0, 20.0);
            let lam = eta.exp();
            grad += &row * (active_w * (c.as_f64() - lam));
            hess += (&row * row.transpose()) * (active_w * lam);
        }
        if grad.norm() < 1e-9 {
            break;
        }
        let Some(step) = hess.lu().solve(&grad) else {
            break;
        };
        let base_obj = poisson_objective(x_mat, counts, w, &beta);
        let mut scale = 1.0;
        while scale >= 1e-4 {
            let cand = &beta + scale * &step;
            if poisson_objective(x_mat, counts, w, &cand) >= base_obj - 1e-12 {
                beta = cand;
                break;
            }
            scale *= 0.5;
        }
    }
    Ok(beta)
}

// [cite:clinical_trials_statistics]
