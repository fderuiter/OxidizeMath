//! Nonlinear parametric curve-fitting dispatch and metrics (`Exponential`, `PowerLaw`,
//! `Logarithmic`, `Logistic`, and `Sinusoidal`).
// [cite:math_commons]

#[cfg(feature = "alloc")]
pub use super::format::format_nonlinear_formula;
use super::lm::{levenberg_marquardt, levenberg_marquardt_filtered};
use super::seeds::{
    WeightedOlsAccum, seed_exponential_shifted, seed_logarithmic, seed_logistic, seed_power_law,
    seed_sinusoidal,
};
use super::{CurveFit, FitModel};
use crate::error::FitError;
use crate::series::PointSeries;
use math_scalars::RealField;

/// Fits a nonlinear [`FitModel`] to `points` on the stack, returning [`FitError`] on failure.
#[cfg_attr(feature = "std", verified_engine::verified)]
pub fn fit_nonlinear<T: RealField, P: PointSeries<T> + ?Sized>(
    points: &P,
    model: FitModel,
    log_mode: bool,
) -> Result<CurveFit<T>, FitError> {
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

fn pack_exponential_coeffs<T: RealField>(a_u: T, b: T, x0: T) -> ([T; 5], usize) {
    if x0 == T::ZERO || (x0.abs() <= T::from_f64(50.0) && (-b * x0).abs() <= T::from_f64(20.0)) {
        let a = a_u * (-b * x0).exp();
        if a.is_finite() && a > T::ZERO {
            return ([a, b, T::ZERO, T::ZERO, T::ZERO], 2);
        }
    }
    ([a_u, b, x0, T::ZERO, T::ZERO], 3)
}

fn fit_exponential<T: RealField, P: PointSeries<T> + ?Sized>(
    points: &P,
    log_mode: bool,
) -> Result<CurveFit<T>, FitError> {
    if log_mode {
        return fit_exponential_log_mode(points);
    }
    let [a0, b0, x0] = seed_exponential_shifted(points)?;
    let lim = T::from_f64(80.0).max(T::MAX_VALUE.ln() - T::TWO);
    let refined = levenberg_marquardt(points, &[a0, b0], |x, p| {
        let u = x - x0;
        let arg = (p[1] * u).clamp(-lim, lim);
        let e = arg.exp();
        (p[0] * e, [e, p[0] * u * e, T::ZERO, T::ZERO])
    })?;
    let (coeffs, coeff_len) = pack_exponential_coeffs(refined[0], refined[1], x0);
    let (r_squared, rmse, valid_n) = compute_nonlinear_metrics(
        points,
        FitModel::Exponential,
        &coeffs[..coeff_len],
        finite_xy::<T>,
    );
    Ok(CurveFit::new(
        FitModel::Exponential,
        coeffs,
        coeff_len,
        r_squared,
        rmse,
        0,
        valid_n,
        false,
        false,
        false,
    ))
}

fn fit_exponential_log_mode<T: RealField, P: PointSeries<T> + ?Sized>(
    points: &P,
) -> Result<CurveFit<T>, FitError> {
    let mut finite_n = 0usize;
    let mut x_min = T::INFINITY;
    for i in 0..points.len() {
        let p = points.point_at(i);
        if p[0].is_finite() && p[1].is_finite() {
            finite_n += 1;
            x_min = x_min.min(p[0]);
        }
    }
    if finite_n < 2 {
        return Err(FitError::InsufficientPoints {
            required: 2,
            actual: finite_n,
        });
    }
    let x0 = if x_min.is_finite() { x_min } else { T::ZERO };
    let mut acc = WeightedOlsAccum::<T>::default();
    for i in 0..points.len() {
        let p = points.point_at(i);
        if p[0].is_finite() && p[1].is_finite() && p[1] > T::ZERO {
            acc.push(p[0] - x0, p[1].ln(), T::ONE);
        }
    }
    if acc.sample_count() < 2 {
        return Err(FitError::InvalidDomain {
            reason: "exponential log-space fit requires at least 2 points with y > 0",
        });
    }
    let (ln_a, b) = acc.solve()?;
    let (coeffs, coeff_len) = pack_exponential_coeffs(ln_a.exp(), b, x0);
    let (r_squared, rmse, valid_n) = compute_nonlinear_metrics(
        points,
        FitModel::Exponential,
        &coeffs[..coeff_len],
        finite_xy::<T>,
    );
    Ok(CurveFit::new(
        FitModel::Exponential,
        coeffs,
        coeff_len,
        r_squared,
        rmse,
        0,
        valid_n,
        false,
        false,
        true,
    ))
}

fn fit_power_law<T: RealField, P: PointSeries<T> + ?Sized>(
    points: &P,
    log_mode: bool,
) -> Result<CurveFit<T>, FitError> {
    if log_mode {
        return fit_power_law_log_mode(points);
    }
    let seed = seed_power_law(points)?;
    let refined = levenberg_marquardt_filtered(points, &seed, positive_x_xy::<T>, |x, p| {
        let xb = x.powf(p[1]);
        let f_hat = p[0] * xb;
        (f_hat, [xb, f_hat * x.ln(), T::ZERO, T::ZERO])
    })?;
    let coeffs = [refined[0], refined[1], T::ZERO, T::ZERO, T::ZERO];
    let (r_squared, rmse, valid_n) =
        compute_nonlinear_metrics(points, FitModel::PowerLaw, &coeffs[..2], positive_x_xy::<T>);
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

fn fit_power_law_log_mode<T: RealField, P: PointSeries<T> + ?Sized>(
    points: &P,
) -> Result<CurveFit<T>, FitError> {
    let mut finite_n = 0usize;
    let mut acc = WeightedOlsAccum::<T>::default();
    for i in 0..points.len() {
        let p = points.point_at(i);
        if !p[0].is_finite() || !p[1].is_finite() {
            continue;
        }
        finite_n += 1;
        if p[0] > T::ZERO && p[1] > T::ZERO {
            acc.push(p[0].ln(), p[1].ln(), T::ONE);
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
    let coeffs = [ln_a.exp(), b, T::ZERO, T::ZERO, T::ZERO];
    let (r_squared, rmse, valid_n) =
        compute_nonlinear_metrics(points, FitModel::PowerLaw, &coeffs[..2], positive_x_xy::<T>);
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

fn fit_logarithmic<T: RealField, P: PointSeries<T> + ?Sized>(
    points: &P,
) -> Result<CurveFit<T>, FitError> {
    let [a, b] = seed_logarithmic(points)?;
    let coeffs = [a, b, T::ZERO, T::ZERO, T::ZERO];
    let (r_squared, rmse, valid_n) = compute_nonlinear_metrics(
        points,
        FitModel::Logarithmic,
        &coeffs[..2],
        positive_x_xy::<T>,
    );
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

fn fit_logistic<T: RealField, P: PointSeries<T> + ?Sized>(
    points: &P,
) -> Result<CurveFit<T>, FitError> {
    let seed = seed_logistic(points)?;
    let lim = T::from_f64(80.0).min(T::MAX_VALUE.ln() - T::TWO);
    let refined = levenberg_marquardt(points, &seed, |x, p| {
        let z = (-p[1] * (x - p[2])).clamp(-lim, lim);
        let ez = z.exp();
        let denom = T::ONE + ez;
        let inv = T::ONE / denom;
        let factor = p[0] * ez * inv * inv;
        (
            p[0] * inv,
            [inv, factor * (x - p[2]), -factor * p[1], T::ZERO],
        )
    })?;
    let coeffs = [refined[0], refined[1], refined[2], T::ZERO, T::ZERO];
    let (r_squared, rmse, valid_n) =
        compute_nonlinear_metrics(points, FitModel::Logistic, &coeffs[..3], finite_xy::<T>);
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

fn min_finite_x<T: RealField, P: PointSeries<T> + ?Sized>(points: &P) -> T {
    let mut x_min = T::INFINITY;
    for i in 0..points.len() {
        let p = points.point_at(i);
        if p[0].is_finite() && p[1].is_finite() {
            x_min = x_min.min(p[0]);
        }
    }
    if x_min.is_finite() { x_min } else { T::ZERO }
}

fn fit_sinusoidal<T: RealField, P: PointSeries<T> + ?Sized>(
    points: &P,
) -> Result<CurveFit<T>, FitError> {
    let seed = seed_sinusoidal(points)?;
    let x0 = min_finite_x(points);
    let mut phase_u = (seed[2] + seed[1] * x0).rem_euclid(T::TAU);
    if phase_u > T::PI {
        phase_u -= T::TAU;
    }
    let shifted_seed = [seed[0], seed[1], phase_u, seed[3]];
    let refined = levenberg_marquardt(points, &shifted_seed, |x, p| {
        let u = x - x0;
        let (s, c) = (p[1] * u + p[2]).sin_cos();
        (p[0] * s + p[3], [s, p[0] * u * c, p[0] * c, T::ONE])
    })?;
    let mut amp = refined[0];
    let mut phase = refined[2] - refined[1] * x0;
    if amp < T::ZERO {
        amp = -amp;
        phase += T::PI;
    }
    phase = phase.rem_euclid(T::TAU);
    if phase > T::PI {
        phase -= T::TAU;
    }
    let coeffs = [amp, refined[1], phase, refined[3], T::ZERO];
    let (r_squared, rmse, valid_n) =
        compute_nonlinear_metrics(points, FitModel::Sinusoidal, &coeffs[..4], finite_xy::<T>);
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
fn finite_xy<T: RealField>(x: T, y: T) -> bool {
    x.is_finite() && y.is_finite()
}

#[inline]
fn positive_x_xy<T: RealField>(x: T, y: T) -> bool {
    x.is_finite() && y.is_finite() && x > T::ZERO
}

fn compute_nonlinear_metrics<T: RealField, P: PointSeries<T> + ?Sized>(
    points: &P,
    model: FitModel,
    params: &[T],
    domain_ok: fn(T, T) -> bool,
) -> (T, T, usize) {
    let mut count = 0usize;
    let mut sum_y = T::ZERO;
    for i in 0..points.len() {
        let p = points.point_at(i);
        if domain_ok(p[0], p[1]) {
            count += 1;
            sum_y += p[1];
        }
    }
    if count == 0 {
        return (T::ZERO, T::NAN, 0);
    }
    let inv_count = T::ONE / T::from_usize(count);
    let mean_y = sum_y * inv_count;
    let mut ss_tot = T::ZERO;
    let mut ss_res = T::ZERO;
    for i in 0..points.len() {
        let p = points.point_at(i);
        if !domain_ok(p[0], p[1]) {
            continue;
        }
        let y_hat = eval_nonlinear_raw(model, params, p[0]);
        if y_hat.is_finite() {
            let diff = p[1] - y_hat;
            let tot = p[1] - mean_y;
            ss_res += diff * diff;
            ss_tot += tot * tot;
        }
    }
    let r_squared = if ss_tot > T::from_f64(1e-15) {
        (T::ONE - ss_res / ss_tot).clamp(T::from_f64(-10.0), T::ONE)
    } else if ss_res <= T::from_f64(1e-12) {
        T::ONE
    } else {
        T::ZERO
    };
    (r_squared, (ss_res * inv_count).sqrt(), count)
}

/// Evaluates nonlinear [`FitModel`] `model` with `params` at horizontal coordinate `x`.
#[inline]
#[must_use]
#[cfg_attr(feature = "std", verified_engine::verified)]
pub fn eval_nonlinear<T: RealField>(model: FitModel, params: &[T], x: T) -> T {
    eval_nonlinear_raw(model, params, x)
}

#[inline]
pub(super) fn eval_nonlinear_raw<T: RealField>(model: FitModel, params: &[T], x: T) -> T {
    match model {
        FitModel::Exponential if params.len() >= 2 => {
            let x0 = params.get(2).copied().unwrap_or(T::ZERO);
            let arg = params[1] * (x - x0);
            let max_ln = T::MAX_VALUE.ln() - T::ONE;
            if arg < T::from_f64(-745.0) {
                T::ZERO
            } else if arg > max_ln {
                T::NAN
            } else {
                params[0] * arg.exp()
            }
        }
        FitModel::PowerLaw if params.len() >= 2 => {
            if x <= T::ZERO {
                T::NAN
            } else {
                params[0] * x.powf(params[1])
            }
        }
        FitModel::Logarithmic if params.len() >= 2 => {
            if x <= T::ZERO {
                T::NAN
            } else {
                params[0] + params[1] * x.ln()
            }
        }
        FitModel::Logistic if params.len() >= 3 => {
            let lim = T::from_f64(80.0).min(T::MAX_VALUE.ln() - T::TWO);
            let z = (-params[1] * (x - params[2])).clamp(-lim, lim);
            params[0] / (T::ONE + z.exp())
        }
        FitModel::Sinusoidal if params.len() >= 4 => {
            params[0] * (params[1] * x + params[2]).sin() + params[3]
        }
        _ => T::NAN,
    }
}
