//! Nonlinear model evaluation, analytical Jacobians, formula formatting, and fit assembly.

use crate::fit::{
    FitResult, FitType, MAX_FIT_PARAMS,
    lm::levenberg_marquardt,
    seeds::{seed_exponential, seed_logarithmic, seed_logistic, seed_power_law, seed_sinusoidal},
};

pub(crate) fn eval_nonlinear_model(fit_type: FitType, coeffs: &[f64], x: f64) -> f64 {
    let out = match fit_type {
        FitType::Exponential => eval_exp(x, coeffs),
        FitType::PowerLaw => eval_power(x, coeffs),
        FitType::Logarithmic => eval_log(x, coeffs),
        FitType::Logistic => eval_logistic(x, coeffs),
        FitType::Sinusoidal => eval_sin(x, coeffs),
        FitType::Linear | FitType::Polynomial(_) | FitType::Custom => None,
    };
    out.unwrap_or(f64::NAN)
}

pub(crate) fn eval_exp(x: f64, c: &[f64]) -> Option<f64> {
    if c.len() < 2 {
        return None;
    }
    let arg = c[1] * x;
    if !arg.is_finite() || arg > 700.0 {
        return None;
    }
    let y = c[0] * arg.exp();
    y.is_finite().then_some(y)
}

fn jac_exp(x: f64, c: &[f64], jac: &mut [f64; MAX_FIT_PARAMS]) -> Option<f64> {
    let arg = c[1] * x;
    if !arg.is_finite() || arg > 700.0 {
        return None;
    }
    let e = arg.exp();
    let y = c[0] * e;
    jac[0] = e;
    jac[1] = y * x;
    y.is_finite().then_some(y)
}

pub(crate) fn eval_power(x: f64, c: &[f64]) -> Option<f64> {
    if c.len() < 2 || x <= 0.0 {
        return None;
    }
    let y = c[0] * x.powf(c[1]);
    y.is_finite().then_some(y)
}

fn jac_power(x: f64, c: &[f64], jac: &mut [f64; MAX_FIT_PARAMS]) -> Option<f64> {
    if x <= 0.0 {
        return None;
    }
    let lx = x.ln();
    let xb = x.powf(c[1]);
    let y = c[0] * xb;
    jac[0] = xb;
    jac[1] = y * lx;
    (y.is_finite() && lx.is_finite()).then_some(y)
}

pub(crate) fn eval_log(x: f64, c: &[f64]) -> Option<f64> {
    if c.len() < 2 || x <= 0.0 {
        return None;
    }
    let y = c[0] * x.ln() + c[1];
    y.is_finite().then_some(y)
}

pub(crate) fn eval_logistic(x: f64, c: &[f64]) -> Option<f64> {
    if c.len() < 3 {
        return None;
    }
    let arg = (-c[2] * x).clamp(-700.0, 700.0);
    let denom = 1.0 + c[1] * arg.exp();
    if !denom.is_finite() || denom.abs() <= 1e-15 {
        return None;
    }
    let y = c[0] / denom;
    y.is_finite().then_some(y)
}

fn jac_logistic(x: f64, c: &[f64], jac: &mut [f64; MAX_FIT_PARAMS]) -> Option<f64> {
    let arg = -c[2] * x;
    if !arg.is_finite() || arg > 700.0 {
        return None;
    }
    let e = arg.exp();
    let denom = 1.0 + c[1] * e;
    if !denom.is_finite() || denom.abs() <= 1e-15 {
        return None;
    }
    let inv_d = 1.0 / denom;
    let y = c[0] * inv_d;
    let y_inv_d = y * inv_d;
    jac[0] = inv_d;
    jac[1] = -y_inv_d * e;
    jac[2] = y_inv_d * c[1] * x * e;
    y.is_finite().then_some(y)
}

pub(crate) fn eval_sin(x: f64, c: &[f64]) -> Option<f64> {
    if c.len() < 4 {
        return None;
    }
    let phase = c[1] * x + c[2];
    if !phase.is_finite() {
        return None;
    }
    let y = c[0] * phase.sin() + c[3];
    y.is_finite().then_some(y)
}

fn jac_sin(x: f64, c: &[f64], jac: &mut [f64; MAX_FIT_PARAMS]) -> Option<f64> {
    let phase = c[1] * x + c[2];
    if !phase.is_finite() {
        return None;
    }
    let (s, cs) = phase.sin_cos();
    let y = c[0] * s + c[3];
    jac[0] = s;
    jac[1] = c[0] * x * cs;
    jac[2] = c[0] * cs;
    jac[3] = 1.0;
    y.is_finite().then_some(y)
}

pub(crate) fn fit_nonlinear(
    points: &[[f64; 2]],
    fit_type: FitType,
    dataset_name: &str,
    log_mode: bool,
) -> Option<FitResult> {
    let requires_pos_x = matches!(fit_type, FitType::PowerLaw | FitType::Logarithmic);
    let valid: Vec<[f64; 2]> = points
        .iter()
        .copied()
        .filter(|p| p[0].is_finite() && p[1].is_finite() && (!requires_pos_x || p[0] > 0.0))
        .collect();
    if valid.len() < min_points_for_model(fit_type) {
        return None;
    }
    let use_log_mode = log_mode && fit_type.supports_log_mode();
    let coeffs = solve_model_parameters(&valid, fit_type, use_log_mode)?;
    let (r_squared, rmse, residuals) = compute_nonlinear_metrics(&valid, fit_type, &coeffs);
    Some(FitResult {
        fit_type,
        formula: format_nonlinear_formula(fit_type, &coeffs),
        coefficients: coeffs,
        r_squared,
        rmse,
        residuals,
        degree: 0,
        dataset_name: dataset_name.to_owned(),
        log_x: false,
        log_y: false,
        log_mode: use_log_mode,
        custom_evaluator: None,
    })
}

const fn min_points_for_model(fit_type: FitType) -> usize {
    match fit_type {
        FitType::Logistic => 3,
        FitType::Sinusoidal => 4,
        _ => 2,
    }
}

fn solve_model_parameters(
    valid: &[[f64; 2]],
    fit_type: FitType,
    use_log_mode: bool,
) -> Option<Vec<f64>> {
    match fit_type {
        FitType::Exponential => {
            let seed = seed_exponential(valid, use_log_mode)?;
            if use_log_mode {
                Some(seed.to_vec())
            } else {
                levenberg_marquardt(valid, &seed, eval_exp, jac_exp).or_else(|| Some(seed.to_vec()))
            }
        }
        FitType::PowerLaw => {
            let seed = seed_power_law(valid, use_log_mode)?;
            if use_log_mode {
                Some(seed.to_vec())
            } else {
                levenberg_marquardt(valid, &seed, eval_power, jac_power)
                    .or_else(|| Some(seed.to_vec()))
            }
        }
        FitType::Logarithmic => seed_logarithmic(valid).map(|s| s.to_vec()),
        FitType::Logistic => {
            let seed = seed_logistic(valid)?;
            levenberg_marquardt(valid, &seed, eval_logistic, jac_logistic)
                .or_else(|| Some(seed.to_vec()))
        }
        FitType::Sinusoidal => {
            let seed = seed_sinusoidal(valid)?;
            let mut params = levenberg_marquardt(valid, &seed, eval_sin, jac_sin)
                .unwrap_or_else(|| seed.to_vec());
            normalize_sinusoidal_params(&mut params);
            Some(params)
        }
        FitType::Linear | FitType::Polynomial(_) | FitType::Custom => None,
    }
}

fn normalize_sinusoidal_params(params: &mut [f64]) {
    if params.len() < 4 {
        return;
    }
    if params[1] < 0.0 {
        params[1] = -params[1];
        params[0] = -params[0];
    }
    if params[0] < 0.0 {
        params[0] = -params[0];
        params[2] += std::f64::consts::PI;
    }
    let two_pi = 2.0 * std::f64::consts::PI;
    params[2] = (params[2] + std::f64::consts::PI).rem_euclid(two_pi) - std::f64::consts::PI;
}

fn compute_nonlinear_metrics(
    valid: &[[f64; 2]],
    fit_type: FitType,
    coeffs: &[f64],
) -> (f64, f64, Vec<[f64; 2]>) {
    let n = valid.len() as f64;
    let y_mean = valid.iter().map(|p| p[1]).sum::<f64>() / n;
    let (mut ss_tot, mut ss_res) = (0.0, 0.0);
    let mut residuals = Vec::with_capacity(valid.len());
    for p in valid {
        let y_hat = eval_nonlinear_model(fit_type, coeffs, p[0]);
        let res = p[1] - y_hat;
        if res.is_finite() {
            residuals.push([p[0], res]);
            ss_res += res * res;
            ss_tot += (p[1] - y_mean).powi(2);
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

fn format_nonlinear_formula(fit_type: FitType, c: &[f64]) -> String {
    match fit_type {
        FitType::Exponential => format!("y = {:.4}·exp({:+.4}x)", c[0], c[1]),
        FitType::PowerLaw => format!("y = {:.4}·x^({:.4})", c[0], c[1]),
        FitType::Logarithmic => {
            let sign = if c[1] < 0.0 { "-" } else { "+" };
            format!("y = {:.4}·ln(x) {sign} {:.4}", c[0], c[1].abs())
        }
        FitType::Logistic => format!("y = {:.4} / (1 + {:.4}·exp({:+.4}x))", c[0], c[1], -c[2]),
        FitType::Sinusoidal => {
            let d_sign = if c[3] < 0.0 { "-" } else { "+" };
            format!(
                "y = {:.4}·sin({:.4}x {:+.4}) {d_sign} {:.4}",
                c[0],
                c[1],
                c[2],
                c[3].abs()
            )
        }
        FitType::Linear | FitType::Polynomial(_) | FitType::Custom => String::new(),
    }
}
