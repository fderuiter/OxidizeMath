//! Callback-driven custom nonlinear regression up to 6 free parameters without external AST crates.

use std::sync::Arc;

use crate::fit::{
    FitResult, FitType, MAX_FIT_PARAMS,
    lm::{levenberg_marquardt_numeric, sum_squared_errors_with},
};

/// Compiled custom 1D regression model $y = f(x; \theta)$ supplied by an external callback.
pub trait CustomFitEvaluator: Send + Sync + std::fmt::Debug {
    /// Ordered free parameter names (`0..=6` parameters).
    fn param_names(&self) -> &[String];

    /// Evaluates $f(x; \theta)$ at `x` for `params[..self.param_names().len()]`.
    fn eval(&self, x: f64, params: &[f64]) -> Option<f64>;

    /// Formats the fitted formula badge string with resolved parameter values.
    fn format_fitted_formula(&self, raw_formula: &str, params: &[f64]) -> String {
        let trimmed = raw_formula.trim();
        let body = trimmed
            .split_once('=')
            .map_or(trimmed, |(_, rhs)| rhs.trim());
        if params.is_empty() {
            format!("y = {body}")
        } else {
            let pairs: Vec<String> = self
                .param_names()
                .iter()
                .zip(params.iter())
                .map(|(n, v)| format!("{n}={v:.4}"))
                .collect();
            format!("y = {body} ({})", pairs.join(", "))
        }
    }
}

/// Cloneable, pointer-comparable handle wrapping a [`CustomFitEvaluator`] inside [`FitResult`].
#[derive(Clone, Debug)]
pub struct CustomEvaluatorHandle(pub Arc<dyn CustomFitEvaluator>);

impl PartialEq for CustomEvaluatorHandle {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

/// Callback signature compiling a user formula string into a [`CustomFitEvaluator`].
pub type CustomFitCallback =
    Arc<dyn Fn(&str) -> Result<Arc<dyn CustomFitEvaluator>, String> + Send + Sync>;

/// Fits a pre-compiled [`CustomFitEvaluator`] (up to [`MAX_FIT_PARAMS`] parameters) to `points`
/// using damped 6-parameter Levenberg-Marquardt.
pub fn fit_custom_evaluator(
    points: &[[f64; 2]],
    evaluator: Arc<dyn CustomFitEvaluator>,
    raw_formula: &str,
    initial_seeds: &[f64],
    dataset_name: &str,
) -> Result<FitResult, String> {
    let valid: Vec<[f64; 2]> = points
        .iter()
        .copied()
        .filter(|p| p[0].is_finite() && p[1].is_finite())
        .collect();
    let k = evaluator.param_names().len();
    if k > MAX_FIT_PARAMS {
        return Err(format!(
            "Custom regression supports at most {MAX_FIT_PARAMS} parameters (got {k})"
        ));
    }
    let min_pts = k.max(2);
    if valid.len() < min_pts {
        return Err(format!(
            "Custom regression requires at least {min_pts} finite points"
        ));
    }
    let coeffs = solve_custom_parameters(&valid, evaluator.as_ref(), initial_seeds, k)?;
    let (r_squared, rmse, residuals) = compute_custom_metrics(&valid, evaluator.as_ref(), &coeffs);
    Ok(FitResult {
        fit_type: FitType::Custom,
        formula: evaluator.format_fitted_formula(raw_formula, &coeffs),
        coefficients: coeffs,
        r_squared,
        rmse,
        residuals,
        degree: 0,
        dataset_name: dataset_name.to_owned(),
        log_x: false,
        log_y: false,
        log_mode: false,
        custom_evaluator: Some(CustomEvaluatorHandle(evaluator)),
    })
}

fn solve_custom_parameters(
    valid: &[[f64; 2]],
    evaluator: &dyn CustomFitEvaluator,
    initial_seeds: &[f64],
    k: usize,
) -> Result<Vec<f64>, String> {
    if k == 0 {
        let eval_fn = |x: f64, p: &[f64]| evaluator.eval(x, p);
        if sum_squared_errors_with(valid, &[], &eval_fn).is_none() {
            return Err("Custom formula produced non-finite values on dataset".to_owned());
        }
        return Ok(Vec::new());
    }

    let candidates = build_candidate_seeds(valid, initial_seeds, k);
    let eval_fn = |x: f64, p: &[f64]| evaluator.eval(x, p);
    let mut best: Option<(f64, Vec<f64>)> = None;

    for seed in candidates {
        if let Some(fitted) = levenberg_marquardt_numeric(valid, &seed, &eval_fn)
            && let Some(sse) = sum_squared_errors_with(valid, &fitted, &eval_fn)
            && best.as_ref().is_none_or(|(best_sse, _)| sse < *best_sse)
        {
            best = Some((sse, fitted));
            if sse <= 1e-20 {
                break;
            }
        }
    }
    best.map(|(_, theta)| theta)
        .ok_or_else(|| "Levenberg-Marquardt failed to evaluate finite residuals".to_owned())
}

fn build_candidate_seeds(valid: &[[f64; 2]], initial_seeds: &[f64], k: usize) -> Vec<Vec<f64>> {
    if !initial_seeds.is_empty() {
        let mut explicit = vec![1.0; k];
        let copy_len = initial_seeds.len().min(k);
        explicit[..copy_len].copy_from_slice(&initial_seeds[..copy_len]);
        return vec![explicit];
    }
    let (mut x_min, mut x_max) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut y_min, mut y_max, mut y_sum) = (f64::INFINITY, f64::NEG_INFINITY, 0.0);
    for &[x, y] in valid {
        x_min = x_min.min(x);
        x_max = x_max.max(x);
        y_min = y_min.min(y);
        y_max = y_max.max(y);
        y_sum += y;
    }
    let x_span = (x_max - x_min).abs().max(1e-3);
    let y_amp = (0.5 * (y_max - y_min)).max(0.5);
    let y_mean = y_sum / (valid.len() as f64);
    let scale_template = [y_amp, 1.0 / x_span, 0.1, y_mean, 0.5, 0.5];
    let mut scaled = vec![1.0; k];
    scaled.copy_from_slice(&scale_template[..k]);
    vec![vec![1.0; k], scaled, vec![0.5; k]]
}

fn compute_custom_metrics(
    valid: &[[f64; 2]],
    evaluator: &dyn CustomFitEvaluator,
    coeffs: &[f64],
) -> (f64, f64, Vec<[f64; 2]>) {
    let y_mean = valid.iter().map(|p| p[1]).sum::<f64>() / (valid.len() as f64);
    let (mut ss_tot, mut ss_res) = (0.0, 0.0);
    let mut residuals = Vec::with_capacity(valid.len());
    for p in valid {
        if let Some(y_hat) = evaluator.eval(p[0], coeffs) {
            let res = p[1] - y_hat;
            if res.is_finite() {
                residuals.push([p[0], res]);
                ss_res += res * res;
                ss_tot += (p[1] - y_mean).powi(2);
            }
        }
    }
    let r2 = if ss_tot < 1e-12 {
        if ss_res < 1e-12 { 1.0 } else { 0.0 }
    } else {
        (1.0 - ss_res / ss_tot).clamp(0.0, 1.0)
    };
    let rmse = if residuals.is_empty() {
        0.0
    } else {
        (ss_res / residuals.len() as f64).sqrt()
    };
    (r2, rmse, residuals)
}
