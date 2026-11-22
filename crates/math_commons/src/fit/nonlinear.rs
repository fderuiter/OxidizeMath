//! Nonlinear parametric curve-fitting dispatch and metrics (`Exponential`, `PowerLaw`,
//! `Logarithmic`, `Logistic`, and `Sinusoidal`).
// [cite:math_commons]

use super::lm::{levenberg_marquardt, levenberg_marquardt_filtered};
use super::seeds::{
    WeightedOlsAccum, seed_exponential, seed_logarithmic, seed_logistic, seed_power_law,
    seed_sinusoidal,
};
use super::{CurveFit, FitModel};
use crate::error::FitError;
use crate::series::PointSeries;

/// Fits a nonlinear [`FitModel`] to `points` on the stack, returning [`FitError`] on failure.
#[verified_engine::verified]
pub fn fit_nonlinear<P: PointSeries + ?Sized>(
    points: &P,
    model: FitModel,
    log_mode: bool,
) -> Result<CurveFit, FitError> {
    match model {
        FitModel::Exponential => fit_exponential(points, log_mode),
        FitModel::PowerLaw => fit_power_law(points, log_mode),
        FitModel::Logarithmic => fit_logarithmic(points),
        FitModel::Logistic => fit_logistic(points),
        FitModel::Sinusoidal => fit_sinusoidal(points),
        FitModel::Linear | FitModel::Quadratic | FitModel::Cubic | FitModel::Quartic => {
            Err(FitError::InvalidDomain {
                reason: "polynomial FitModel passed to fit_nonlinear",
            })
        }
    }
}

#[verified_engine::verified]
fn fit_exponential<P: PointSeries + ?Sized>(
    points: &P,
    log_mode: bool,
) -> Result<CurveFit, FitError> {
    if log_mode {
        return fit_exponential_log_mode(points);
    }
    let seed = seed_exponential(points)?;
    let refined = levenberg_marquardt(points, &seed, |x, p| {
        let e = (p[1] * x).exp();
        (p[0] * e, [e, p[0] * x * e, 0.0, 0.0])
    })?;
    let coeffs = [refined[0], refined[1], 0.0, 0.0, 0.0];
    let (r_squared, rmse, valid_n) =
        compute_nonlinear_metrics(points, FitModel::Exponential, &coeffs[..2], finite_xy);
    Ok(CurveFit::new(
        FitModel::Exponential,
        coeffs,
        2,
        r_squared,
        rmse,
        0,
        valid_n,
        false,
        false,
        false,
    ))
}

#[verified_engine::verified]
fn fit_exponential_log_mode<P: PointSeries + ?Sized>(points: &P) -> Result<CurveFit, FitError> {
    let mut finite_n = 0usize;
    let mut acc = WeightedOlsAccum::default();
    for i in 0..points.len() {
        let p = points.point_at(i);
        if !p[0].is_finite() || !p[1].is_finite() {
            continue;
        }
        finite_n += 1;
        if p[1] > 0.0 {
            acc.push(p[0], p[1].ln(), 1.0);
        }
    }
    if finite_n < 2 {
        return Err(FitError::InsufficientPoints {
            required: 2,
            actual: finite_n,
        });
    }
    if acc.sample_count() < 2 {
        return Err(FitError::InvalidDomain {
            reason: "exponential log-space fit requires at least 2 points with y > 0",
        });
    }
    let (ln_a, b) = acc.solve()?;
    let a = ln_a.exp();
    let coeffs = [a, b, 0.0, 0.0, 0.0];
    let (r_squared, rmse, valid_n) =
        compute_nonlinear_metrics(points, FitModel::Exponential, &coeffs[..2], finite_xy);
    Ok(CurveFit::new(
        FitModel::Exponential,
        coeffs,
        2,
        r_squared,
        rmse,
        0,
        valid_n,
        false,
        false,
        true,
    ))
}

#[verified_engine::verified]
fn fit_power_law<P: PointSeries + ?Sized>(
    points: &P,
    log_mode: bool,
) -> Result<CurveFit, FitError> {
    if log_mode {
        return fit_power_law_log_mode(points);
    }
    let seed = seed_power_law(points)?;
    let refined = levenberg_marquardt_filtered(points, &seed, positive_x_xy, |x, p| {
        let xb = x.powf(p[1]);
        let f_hat = p[0] * xb;
        (f_hat, [xb, f_hat * x.ln(), 0.0, 0.0])
    })?;
    let coeffs = [refined[0], refined[1], 0.0, 0.0, 0.0];
    let (r_squared, rmse, valid_n) =
        compute_nonlinear_metrics(points, FitModel::PowerLaw, &coeffs[..2], positive_x_xy);
    Ok(CurveFit::new(
        FitModel::PowerLaw,
        coeffs,
        2,
        r_squared,
        rmse,
        0,
        valid_n,
        false,
        false,
        false,
    ))
}

#[verified_engine::verified]
fn fit_power_law_log_mode<P: PointSeries + ?Sized>(points: &P) -> Result<CurveFit, FitError> {
    let mut finite_n = 0usize;
    let mut acc = WeightedOlsAccum::default();
    for i in 0..points.len() {
        let p = points.point_at(i);
        if !p[0].is_finite() || !p[1].is_finite() {
            continue;
        }
        finite_n += 1;
        if p[0] > 0.0 && p[1] > 0.0 {
            acc.push(p[0].ln(), p[1].ln(), 1.0);
        }
    }
    if finite_n < 2 {
        return Err(FitError::InsufficientPoints {
            required: 2,
            actual: finite_n,
        });
    }
    if acc.sample_count() < 2 {
        return Err(FitError::InvalidDomain {
            reason: "power-law log-space fit requires at least 2 points with x > 0 and y > 0",
        });
    }
    let (ln_a, b) = acc.solve()?;
    let coeffs = [ln_a.exp(), b, 0.0, 0.0, 0.0];
    let (r_squared, rmse, valid_n) =
        compute_nonlinear_metrics(points, FitModel::PowerLaw, &coeffs[..2], positive_x_xy);
    Ok(CurveFit::new(
        FitModel::PowerLaw,
        coeffs,
        2,
        r_squared,
        rmse,
        0,
        valid_n,
        false,
        false,
        true,
    ))
}

#[verified_engine::verified]
fn fit_logarithmic<P: PointSeries + ?Sized>(points: &P) -> Result<CurveFit, FitError> {
    let [a, b] = seed_logarithmic(points)?;
    let coeffs = [a, b, 0.0, 0.0, 0.0];
    let (r_squared, rmse, valid_n) =
        compute_nonlinear_metrics(points, FitModel::Logarithmic, &coeffs[..2], positive_x_xy);
    Ok(CurveFit::new(
        FitModel::Logarithmic,
        coeffs,
        2,
        r_squared,
        rmse,
        0,
        valid_n,
        false,
        false,
        false,
    ))
}

#[verified_engine::verified]
fn fit_logistic<P: PointSeries + ?Sized>(points: &P) -> Result<CurveFit, FitError> {
    let seed = seed_logistic(points)?;
    let refined = levenberg_marquardt(points, &seed, |x, p| {
        let z = (-p[1] * (x - p[2])).clamp(-500.0, 500.0);
        let ez = z.exp();
        let denom = 1.0 + ez;
        let inv = 1.0 / denom;
        let factor = p[0] * ez * inv * inv;
        (p[0] * inv, [inv, factor * (x - p[2]), -factor * p[1], 0.0])
    })?;
    let coeffs = [refined[0], refined[1], refined[2], 0.0, 0.0];
    let (r_squared, rmse, valid_n) =
        compute_nonlinear_metrics(points, FitModel::Logistic, &coeffs[..3], finite_xy);
    Ok(CurveFit::new(
        FitModel::Logistic,
        coeffs,
        3,
        r_squared,
        rmse,
        0,
        valid_n,
        false,
        false,
        false,
    ))
}

#[verified_engine::verified]
fn fit_sinusoidal<P: PointSeries + ?Sized>(points: &P) -> Result<CurveFit, FitError> {
    let seed = seed_sinusoidal(points)?;
    let refined = levenberg_marquardt(points, &seed, |x, p| {
        let arg = p[1] * x + p[2];
        let s = arg.sin();
        let c = arg.cos();
        (p[0] * s + p[3], [s, p[0] * x * c, p[0] * c, 1.0])
    })?;
    let mut amp = refined[0];
    let mut phase = refined[2];
    if amp < 0.0 {
        amp = -amp;
        phase += std::f64::consts::PI;
    }
    phase = phase.rem_euclid(std::f64::consts::TAU);
    if phase > std::f64::consts::PI {
        phase -= std::f64::consts::TAU;
    }
    let coeffs = [amp, refined[1], phase, refined[3], 0.0];
    let (r_squared, rmse, valid_n) =
        compute_nonlinear_metrics(points, FitModel::Sinusoidal, &coeffs[..4], finite_xy);
    Ok(CurveFit::new(
        FitModel::Sinusoidal,
        coeffs,
        4,
        r_squared,
        rmse,
        0,
        valid_n,
        false,
        false,
        false,
    ))
}

#[inline]
#[verified_engine::verified]
fn finite_xy(x: f64, y: f64) -> bool {
    x.is_finite() && y.is_finite()
}

#[inline]
#[verified_engine::verified]
fn positive_x_xy(x: f64, y: f64) -> bool {
    x.is_finite() && y.is_finite() && x > 0.0
}

#[verified_engine::verified]
fn compute_nonlinear_metrics<P: PointSeries + ?Sized>(
    points: &P,
    model: FitModel,
    params: &[f64],
    domain_ok: fn(f64, f64) -> bool,
) -> (f64, f64, usize) {
    let mut count = 0usize;
    let mut sum_y = 0.0_f64;
    for i in 0..points.len() {
        let p = points.point_at(i);
        if domain_ok(p[0], p[1]) {
            count += 1;
            sum_y += p[1];
        }
    }
    if count == 0 {
        return (0.0, f64::NAN, 0);
    }
    let mean_y = sum_y / (count as f64);
    let mut ss_tot = 0.0_f64;
    let mut ss_res = 0.0_f64;
    for i in 0..points.len() {
        let p = points.point_at(i);
        if !domain_ok(p[0], p[1]) {
            continue;
        }
        let y_hat = eval_nonlinear(model, params, p[0]);
        if y_hat.is_finite() {
            let diff = p[1] - y_hat;
            let tot = p[1] - mean_y;
            ss_res += diff * diff;
            ss_tot += tot * tot;
        }
    }
    let r_squared = if ss_tot > 1e-15 {
        (1.0 - ss_res / ss_tot).clamp(-10.0, 1.0)
    } else if ss_res <= 1e-12 {
        1.0
    } else {
        0.0
    };
    (r_squared, (ss_res / (count as f64)).sqrt(), count)
}

/// Evaluates nonlinear [`FitModel`] `model` with `params` at horizontal coordinate `x`.
#[inline]
#[must_use]
#[verified_engine::verified]
pub fn eval_nonlinear(model: FitModel, params: &[f64], x: f64) -> f64 {
    match model {
        FitModel::Exponential if params.len() >= 2 => params[0] * (params[1] * x).exp(),
        FitModel::PowerLaw if params.len() >= 2 => {
            if x <= 0.0 {
                f64::NAN
            } else {
                params[0] * x.powf(params[1])
            }
        }
        FitModel::Logarithmic if params.len() >= 2 => {
            if x <= 0.0 {
                f64::NAN
            } else {
                params[0] + params[1] * x.ln()
            }
        }
        FitModel::Logistic if params.len() >= 3 => {
            let z = (-params[1] * (x - params[2])).clamp(-500.0, 500.0);
            params[0] / (1.0 + z.exp())
        }
        FitModel::Sinusoidal if params.len() >= 4 => {
            params[0] * (params[1] * x + params[2]).sin() + params[3]
        }
        _ => f64::NAN,
    }
}

fn fmt_num(val: f64) -> String {
    let abs = val.abs();
    if abs != 0.0 && !(1e-3..1e4).contains(&abs) {
        format!("{val:.3e}")
    } else {
        format!("{val:.4}")
    }
}

/// Formats a human-readable equation string for nonlinear [`FitModel`] presets.
#[must_use]
pub fn format_nonlinear_formula(model: FitModel, params: &[f64]) -> String {
    match model {
        FitModel::Exponential if params.len() >= 2 => {
            format!("y = {}·exp({}·x)", fmt_num(params[0]), fmt_num(params[1]))
        }
        FitModel::PowerLaw if params.len() >= 2 => {
            format!("y = {}·x^({})", fmt_num(params[0]), fmt_num(params[1]))
        }
        FitModel::Logarithmic if params.len() >= 2 => {
            let sign = if params[1] >= 0.0 { "+" } else { "-" };
            format!(
                "y = {} {} {}·ln(x)",
                fmt_num(params[0]),
                sign,
                fmt_num(params[1].abs())
            )
        }
        FitModel::Logistic if params.len() >= 3 => {
            let sign = if params[2] >= 0.0 { "-" } else { "+" };
            format!(
                "y = {} / (1 + exp(-{}·(x {} {})))",
                fmt_num(params[0]),
                fmt_num(params[1]),
                sign,
                fmt_num(params[2].abs())
            )
        }
        FitModel::Sinusoidal if params.len() >= 4 => {
            let c_sign = if params[2] >= 0.0 { "+" } else { "-" };
            let d_sign = if params[3] >= 0.0 { "+" } else { "-" };
            format!(
                "y = {}·sin({}·x {} {}) {} {}",
                fmt_num(params[0]),
                fmt_num(params[1]),
                c_sign,
                fmt_num(params[2].abs()),
                d_sign,
                fmt_num(params[3].abs())
            )
        }
        _ => "y = NaN".to_string(),
    }
}
