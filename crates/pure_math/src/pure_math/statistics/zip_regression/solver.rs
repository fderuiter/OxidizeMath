//! Lambert (1992) Expectation-Maximization (EM) + damped Iteratively Reweighted Least Squares
//! (IRLS) solver for [`ZipRegression`], with safeguarded observed-information polish.

use super::link_functions::LogitLink;
use super::regression::{ZipRegression, simple_zip_fit};
use crate::error::ZipError;
use engine_prob::ln_gamma;
use nalgebra::{DMatrix, DVector};

const MIN_ETA: f64 = -25.0;
const MAX_ETA: f64 = 20.0;
const MAX_ABS_GAMMA: f64 = 35.0;
const RIDGE_NUGGET: f64 = 1.0e-9;

/// Hyperparameters controlling [`ZipRegression::fit_with_config`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ZipFitConfig {
    /// Maximum number of outer EM + IRLS iterations (`>= 1`).
    pub max_iterations: usize,
    /// Maximum inner damped IRLS steps per M-step (`>= 1`).
    pub irls_steps: usize,
    /// Convergence tolerance on log-likelihood and parameter infinity-norm change.
    pub tolerance: f64,
}

impl Default for ZipFitConfig {
    fn default() -> Self {
        Self {
            max_iterations: 200,
            irls_steps: 6,
            tolerance: 1.0e-9,
        }
    }
}

/// Fitted maximum-likelihood parameters and diagnostics produced by [`ZipRegression::fit`].
#[derive(Debug, Clone, PartialEq)]
pub struct ZipFitResult {
    /// Fitted count-model coefficient vector $\hat{\beta}$ ($\ln \lambda = X \hat{\beta}$).
    pub beta: DVector<f64>,
    /// Fitted zero-inflation coefficient vector $\hat{\alpha}$ ($\operatorname{logit}(\rho) = Z \hat{\alpha}$).
    pub alpha: DVector<f64>,
    /// Maximized observed-data log-likelihood $\ell(\hat{\beta}, \hat{\alpha})$.
    pub log_likelihood: f64,
    /// Number of outer EM + IRLS iterations executed.
    pub iterations: usize,
    /// Whether convergence criteria were satisfied within `max_iterations`.
    pub converged: bool,
    /// Posterior structural-zero probabilities $\hat{w}_i = P(\text{structural zero} \mid y_i, x_i, z_i)$.
    pub responsibilities: Vec<f64>,
}

#[inline]
fn log1p_exp(x: f64) -> f64 {
    if x > 0.0 {
        x + (-x).exp().ln_1p()
    } else {
        x.exp().ln_1p()
    }
}

fn stable_zip_obs_ll(y: u32, eta: f64, gamma: f64) -> f64 {
    if !eta.is_finite() || !gamma.is_finite() || !(MIN_ETA..=MAX_ETA).contains(&eta) {
        return f64::NEG_INFINITY;
    }
    let lambda = eta.exp();
    let log1p_exp_gamma = log1p_exp(gamma);
    if y == 0 {
        -lambda + log1p_exp(gamma + lambda) - log1p_exp_gamma
    } else {
        let yf = y as f64;
        -log1p_exp_gamma + yf * eta - lambda - ln_gamma(yf + 1.0)
    }
}

fn total_observed_ll(model: &ZipRegression, beta: &DVector<f64>, alpha: &DVector<f64>) -> f64 {
    let eta_vec = &model.x_matrix * beta;
    let gamma_vec = &model.z_matrix * alpha;
    let mut total = 0.0;
    for i in 0..model.counts.len() {
        let term = stable_zip_obs_ll(model.counts[i].value(), eta_vec[i], gamma_vec[i]);
        if !term.is_finite() {
            return f64::NEG_INFINITY;
        }
        total += term;
    }
    total
}

fn compute_e_step_responsibilities(
    model: &ZipRegression,
    beta: &DVector<f64>,
    alpha: &DVector<f64>,
    w_out: &mut [f64],
) {
    let eta_vec = &model.x_matrix * beta;
    let gamma_vec = &model.z_matrix * alpha;
    for i in 0..model.counts.len() {
        if model.counts[i].value() > 0 {
            w_out[i] = 0.0;
        } else {
            let lam = eta_vec[i].clamp(MIN_ETA, MAX_ETA).exp();
            w_out[i] = LogitLink::link(gamma_vec[i] + lam);
        }
    }
}

fn poisson_q_objective(x: &DMatrix<f64>, y: &[f64], active_w: &[f64], beta: &DVector<f64>) -> f64 {
    let eta = x * beta;
    let mut sum = 0.0;
    for i in 0..y.len() {
        let e = eta[i];
        if !e.is_finite() || !(MIN_ETA..=MAX_ETA).contains(&e) {
            return f64::NEG_INFINITY;
        }
        sum += active_w[i] * (y[i] * e - e.exp());
    }
    sum
}

fn irls_poisson_m_step(
    x: &DMatrix<f64>,
    y: &[f64],
    w_zero: &[f64],
    beta: &mut DVector<f64>,
    steps: usize,
) {
    let n = y.len();
    let p = x.ncols();
    let active_w: Vec<f64> = w_zero.iter().map(|&w| (1.0 - w).max(0.0)).collect();
    let mut cur_obj = poisson_q_objective(x, y, &active_w, beta);

    for _ in 0..steps {
        let eta = x * &*beta;
        let mut grad = DVector::zeros(p);
        let mut fisher = DMatrix::from_diagonal_element(p, p, RIDGE_NUGGET);
        for i in 0..n {
            let vi = active_w[i];
            if vi <= 0.0 {
                continue;
            }
            let lam = eta[i].clamp(MIN_ETA, MAX_ETA).exp();
            let residual = vi * (y[i] - lam);
            let weight = (vi * lam).max(1.0e-12);
            let row = x.row(i);
            for r in 0..p {
                grad[r] += row[r] * residual;
                for c in 0..=r {
                    let add = weight * row[r] * row[c];
                    fisher[(r, c)] += add;
                    if r != c {
                        fisher[(c, r)] += add;
                    }
                }
            }
        }
        let Some(delta) = fisher.lu().solve(&grad) else {
            break;
        };
        if delta.amax() < 1.0e-12 {
            break;
        }
        let mut step = 1.0;
        while step >= 1.0e-6 {
            let cand = &*beta + &delta * step;
            let cand_obj = poisson_q_objective(x, y, &active_w, &cand);
            if cand_obj.is_finite() && cand_obj >= cur_obj - 1.0e-12 {
                *beta = cand;
                cur_obj = cand_obj;
                break;
            }
            step *= 0.5;
        }
    }
}

fn logistic_q_objective(z: &DMatrix<f64>, w_zero: &[f64], alpha: &DVector<f64>) -> f64 {
    let gamma = z * alpha;
    let mut sum = 0.0;
    for i in 0..w_zero.len() {
        let g = gamma[i];
        if !g.is_finite() || g.abs() > MAX_ABS_GAMMA {
            return f64::NEG_INFINITY;
        }
        sum += w_zero[i] * g - log1p_exp(g);
    }
    sum
}

fn irls_logistic_m_step(z: &DMatrix<f64>, w_zero: &[f64], alpha: &mut DVector<f64>, steps: usize) {
    let n = w_zero.len();
    let q = z.ncols();
    let mut cur_obj = logistic_q_objective(z, w_zero, alpha);

    for _ in 0..steps {
        let gamma = z * &*alpha;
        let mut grad = DVector::zeros(q);
        let mut fisher = DMatrix::from_diagonal_element(q, q, RIDGE_NUGGET);
        for i in 0..n {
            let rho = LogitLink::link(gamma[i]);
            let residual = w_zero[i] - rho;
            let weight = (rho * (1.0 - rho)).max(1.0e-10);
            let row = z.row(i);
            for r in 0..q {
                grad[r] += row[r] * residual;
                for c in 0..=r {
                    let add = weight * row[r] * row[c];
                    fisher[(r, c)] += add;
                    if r != c {
                        fisher[(c, r)] += add;
                    }
                }
            }
        }
        let Some(delta) = fisher.lu().solve(&grad) else {
            break;
        };
        if delta.amax() < 1.0e-12 {
            break;
        }
        let mut step = 1.0;
        while step >= 1.0e-6 {
            let cand = &*alpha + &delta * step;
            let cand_obj = logistic_q_objective(z, w_zero, &cand);
            if cand_obj.is_finite() && cand_obj >= cur_obj - 1.0e-12 {
                *alpha = cand;
                cur_obj = cand_obj;
                break;
            }
            step *= 0.5;
        }
    }
}

fn assemble_zip_newton_system(
    model: &ZipRegression,
    beta: &DVector<f64>,
    alpha: &DVector<f64>,
) -> (DVector<f64>, DMatrix<f64>) {
    let (p, q) = (model.x_matrix.ncols(), model.z_matrix.ncols());
    let eta = &model.x_matrix * beta;
    let gamma = &model.z_matrix * alpha;
    let mut grad = DVector::<f64>::zeros(p + q);
    let mut neg_hess = DMatrix::<f64>::zeros(p + q, p + q);

    for i in 0..model.counts.len() {
        let yi = model.counts[i].value();
        let lam = eta[i].clamp(MIN_ETA, MAX_ETA).exp();
        let rho = LogitLink::link(gamma[i]);
        let (g_eta, g_gam, nh_ee, nh_gg, nh_eg) = if yi > 0 {
            ((yi as f64) - lam, -rho, lam, rho * (1.0 - rho), 0.0)
        } else {
            let w = LogitLink::link(gamma[i] + lam);
            let w_var = w * (1.0 - w);
            (
                -(1.0 - w) * lam,
                w - rho,
                (1.0 - w) * lam - w_var * lam * lam,
                rho * (1.0 - rho) - w_var,
                -w_var * lam,
            )
        };
        let (x_row, z_row) = (model.x_matrix.row(i), model.z_matrix.row(i));
        for r in 0..p {
            grad[r] += x_row[r] * g_eta;
            for c in 0..p {
                neg_hess[(r, c)] += nh_ee * x_row[r] * x_row[c];
            }
            for c in 0..q {
                let cross = nh_eg * x_row[r] * z_row[c];
                neg_hess[(r, p + c)] += cross;
                neg_hess[(p + c, r)] += cross;
            }
        }
        for r in 0..q {
            grad[p + r] += z_row[r] * g_gam;
            for c in 0..q {
                neg_hess[(p + r, p + c)] += nh_gg * z_row[r] * z_row[c];
            }
        }
    }
    (grad, neg_hess)
}

fn try_damped_newton_polish(
    model: &ZipRegression,
    beta: &mut DVector<f64>,
    alpha: &mut DVector<f64>,
    cur_ll: f64,
) -> Option<f64> {
    let (p, q) = (model.x_matrix.ncols(), model.z_matrix.ncols());
    let dim = p + q;
    let (grad, neg_hess) = assemble_zip_newton_system(model, beta, alpha);
    for &damp in &[1.0e-8, 1.0e-4, 1.0e-1] {
        let mut sys = neg_hess.clone();
        for d in 0..dim {
            sys[(d, d)] = sys[(d, d)].max(1.0e-8) + damp;
        }
        let Some(step_vec) = sys.lu().solve(&grad) else {
            continue;
        };
        let d_beta = step_vec.rows(0, p).into_owned();
        let d_alpha = step_vec.rows(p, q).into_owned();
        let mut scale = 1.0;
        while scale >= 0.0625 {
            let (cand_b, cand_a) = (&*beta + &d_beta * scale, &*alpha + &d_alpha * scale);
            let cand_ll = total_observed_ll(model, &cand_b, &cand_a);
            if cand_ll.is_finite() && cand_ll > cur_ll + 1.0e-12 {
                *beta = cand_b;
                *alpha = cand_a;
                return Some(cand_ll);
            }
            scale *= 0.5;
        }
    }
    None
}

impl ZipRegression {
    /// Fits the ZIP regression model via Lambert (1992) EM + damped IRLS using default settings.
    #[verified_engine::verified]
    pub fn fit(&self) -> Result<ZipFitResult, ZipError> {
        self.fit_with_config(&ZipFitConfig::default())
    }

    /// Fits the ZIP regression model via EM + damped IRLS over `x_matrix` and `z_matrix` without
    /// clamping zero-inflation probabilities $\rho_i$ at `0.95`.
    #[verified_engine::verified]
    pub fn fit_with_config(&self, config: &ZipFitConfig) -> Result<ZipFitResult, ZipError> {
        if config.max_iterations == 0 || !config.tolerance.is_finite() || config.tolerance <= 0.0 {
            return Err(ZipError::ConvergenceFailed {
                iterations: 0,
                reason: "invalid ZipFitConfig parameters".to_string(),
            });
        }
        let (p, q) = (self.x_matrix.ncols(), self.z_matrix.ncols());
        let y_f64: Vec<f64> = self.counts.iter().map(super::core::Count::as_f64).collect();

        let mom = simple_zip_fit(&self.counts)?;
        let init_w0 = LogitLink::link(
            LogitLink::inverse_link(mom.rho.value().clamp(1.0e-6, 1.0 - 1.0e-6))
                + mom.lambda.value(),
        );
        let mut w_resp: Vec<f64> = self
            .counts
            .iter()
            .map(|c| if c.value() == 0 { init_w0 } else { 0.0 })
            .collect();

        let (mut beta, mut alpha) = (DVector::zeros(p), DVector::zeros(q));
        let irls_steps = config.irls_steps.max(1);
        irls_poisson_m_step(&self.x_matrix, &y_f64, &w_resp, &mut beta, irls_steps);
        irls_logistic_m_step(&self.z_matrix, &w_resp, &mut alpha, irls_steps);

        let mut cur_ll = total_observed_ll(self, &beta, &alpha);
        let param_tol = config.tolerance.sqrt().max(1.0e-8);
        let (mut converged, mut iterations) = (false, 0);

        for iter in 0..config.max_iterations {
            let (prev_beta, prev_alpha, prev_ll) = (beta.clone(), alpha.clone(), cur_ll);

            compute_e_step_responsibilities(self, &beta, &alpha, &mut w_resp);
            irls_poisson_m_step(&self.x_matrix, &y_f64, &w_resp, &mut beta, irls_steps);
            irls_logistic_m_step(&self.z_matrix, &w_resp, &mut alpha, irls_steps);
            let em_ll = total_observed_ll(self, &beta, &alpha);
            if em_ll.is_finite() && em_ll >= prev_ll - 1.0e-10 {
                cur_ll = em_ll;
            } else {
                beta = prev_beta.clone();
                alpha = prev_alpha.clone();
            }

            if iter >= 3
                && let Some(polished_ll) =
                    try_damped_newton_polish(self, &mut beta, &mut alpha, cur_ll)
            {
                cur_ll = polished_ll;
            }

            iterations = iter + 1;
            let step_inf = (&beta - &prev_beta)
                .amax()
                .max((&alpha - &prev_alpha).amax());
            if (cur_ll - prev_ll).abs() <= config.tolerance && step_inf <= param_tol {
                converged = true;
                break;
            }
        }

        compute_e_step_responsibilities(self, &beta, &alpha, &mut w_resp);
        if !cur_ll.is_finite() {
            return Err(ZipError::ConvergenceFailed {
                iterations,
                reason: "non-finite log-likelihood encountered".to_string(),
            });
        }
        Ok(ZipFitResult {
            beta,
            alpha,
            log_likelihood: cur_ll,
            iterations,
            converged,
            responsibilities: w_resp,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::distribution::ZipDistribution;
    use super::*;
    use oxidize_core::rng::OxidizeRng;

    #[test]
    #[verified_engine::verified]
    fn test_zip_fit_recovers_synthetic_beta_and_alpha() {
        let true_beta = DVector::from_vec(vec![1.1, 0.55]);
        let true_alpha = DVector::from_vec(vec![-0.4, 1.35]);
        let n = 1_500_usize;
        let mut x_mat = DMatrix::zeros(n, 2);
        let mut z_mat = DMatrix::zeros(n, 2);
        let mut counts = Vec::with_capacity(n);
        let mut rng = OxidizeRng::new(0x517_2026);

        for i in 0..n {
            let (x1, z1) = (
                -1.5 + 3.0 * (i as f64) / (n as f64),
                ((i * 7 % 29) as f64 - 14.0) / 7.0,
            );
            x_mat[(i, 0)] = 1.0;
            x_mat[(i, 1)] = x1;
            z_mat[(i, 0)] = 1.0;
            z_mat[(i, 1)] = z1;
            let params = ZipRegression::predict(
                &DVector::from_vec(vec![1.0, x1]),
                &DVector::from_vec(vec![1.0, z1]),
                &true_beta,
                &true_alpha,
            )
            .unwrap();
            counts.push(ZipDistribution::new(params).sample(&mut rng));
        }

        let model = ZipRegression::new(counts, x_mat, z_mat).unwrap();
        let fit = model.fit().unwrap();
        assert!(fit.converged);
        assert!((fit.beta[0] - true_beta[0]).abs() < 0.12);
        assert!((fit.beta[1] - true_beta[1]).abs() < 0.08);
        assert!((fit.alpha[0] - true_alpha[0]).abs() < 0.18);
        assert!((fit.alpha[1] - true_alpha[1]).abs() < 0.18);
    }

    #[test]
    #[verified_engine::verified]
    fn test_zip_fit_high_zero_inflation_above_095_unclamped() {
        // True rho = expit(3.5) ≈ 0.9707 > 0.95!
        let true_beta = DVector::from_vec(vec![1.6]);
        let true_alpha = DVector::from_vec(vec![3.5]);
        let n = 2_000_usize;
        let x_mat = DMatrix::from_element(n, 1, 1.0);
        let z_mat = DMatrix::from_element(n, 1, 1.0);
        let mut counts = Vec::with_capacity(n);
        let mut rng = OxidizeRng::new(0x9707_CAFE);
        let x_row = DVector::from_vec(vec![1.0]);
        let z_row = DVector::from_vec(vec![1.0]);
        let true_params = ZipRegression::predict(&x_row, &z_row, &true_beta, &true_alpha).unwrap();
        assert!(true_params.rho.value() > 0.97);

        let dist = ZipDistribution::new(true_params);
        for _ in 0..n {
            counts.push(dist.sample(&mut rng));
        }

        let mom = simple_zip_fit(&counts).unwrap();
        assert!(mom.rho.value() > 0.955, "got {}", mom.rho.value());

        let model = ZipRegression::new(counts, x_mat, z_mat).unwrap();
        let fit = model.fit().unwrap();
        assert!(fit.converged);
        let fitted_params = ZipRegression::predict(&x_row, &z_row, &fit.beta, &fit.alpha).unwrap();
        assert!(
            fitted_params.rho.value() > 0.955,
            "got {}",
            fitted_params.rho.value()
        );
        assert!((fitted_params.rho.value() - true_params.rho.value()).abs() < 0.02);
        assert!((fitted_params.lambda.value() - true_params.lambda.value()).abs() < 0.8);
    }
}
