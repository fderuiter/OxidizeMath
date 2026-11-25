//! Stack-allocated Levenberg-Marquardt calibration for [`crate::applied::battery_degradation::ElectrochemicalAgingModel`].
// [cite:battery_degradation]

use super::model::{ElectrochemicalAgingModel, MAX_EXPONENT_Z, MIN_EXPONENT_Z};
use super::types::DepthOfDischarge;
use math_commons::error::FitError;
use math_commons::fit::{MAX_FIT_PARAMS, levenberg_marquardt_ad, sum_squared_errors_numeric};
use math_commons::series::PointSeries;
use math_scalars::{DualVec, Scalar};

const Z_SPAN: f64 = MAX_EXPONENT_Z - MIN_EXPONENT_Z;

/// Result of calibrating an [`ElectrochemicalAgingModel`] to lab `(cycles, capacity)` measurements.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CalibratedAgingFit {
    /// Calibrated aging model with updated prefactor $\alpha$ and stretched exponent $z$.
    pub model: ElectrochemicalAgingModel,
    /// Fitted effective cycle life $N_{70}$ at the calibration test conditions.
    pub fitted_n70_cycles: f64,
    /// Fitted stretched exponent $z \in [0.1, 2.0]$.
    pub fitted_exponent_z: f64,
    /// Root-mean-square state-of-health residual over finite `(cycles, capacity)` observations.
    pub rmse: f64,
    /// Coefficient of determination $R^2$ clamped to $(-\infty, 1.0]$.
    pub r_squared: f64,
    /// Count of finite `(cycles, capacity)` observations used in calibration.
    pub sample_count: usize,
}

#[inline]
fn decode_theta(params: &[f64]) -> Option<(f64, f64)> {
    if params.len() < 2 || !params[0].is_finite() || !params[1].is_finite() {
        return None;
    }
    let ln_n70 = params[0].clamp(-2.0, 18.0);
    let logit_z = params[1].clamp(-20.0, 20.0);
    let n70_eff = ln_n70.exp();
    let z = MIN_EXPONENT_Z + Z_SPAN / (1.0 + (-logit_z).exp());
    Some((n70_eff, z))
}

type FitDual = DualVec<f64, MAX_FIT_PARAMS>;

#[inline]
fn clamp_dual_param(p: FitDual, lo: f64, hi: f64) -> FitDual {
    if p.val < lo {
        FitDual::from_f64(lo)
    } else if p.val > hi {
        FitDual::from_f64(hi)
    } else {
        p
    }
}

#[inline]
fn eval_capacity_ad(cycles: f64, params: &[FitDual]) -> Option<FitDual> {
    if !cycles.is_finite() || cycles < 0.0 || params.len() < 2 {
        return None;
    }
    if !params[0].val.is_finite() || !params[1].val.is_finite() {
        return None;
    }
    if cycles == 0.0 {
        return Some(FitDual::ONE);
    }
    let ln_n70 = clamp_dual_param(params[0], -2.0, 18.0);
    let logit_z = clamp_dual_param(params[1], -20.0, 20.0);
    let n70_eff = ln_n70.exp();
    let z = FitDual::from_f64(MIN_EXPONENT_Z)
        + FitDual::from_f64(Z_SPAN) / (FitDual::ONE + (-logit_z).exp());
    let ratio = FitDual::from_f64(cycles) / n70_eff;
    let raw_loss = FitDual::from_f64(0.3) * ratio.powf(z);
    let loss = clamp_dual_param(raw_loss, 0.0, 1.0);
    Some(FitDual::ONE - loss)
}

#[inline]
fn eval_capacity_at(cycles: f64, params: &[f64]) -> Option<f64> {
    if !cycles.is_finite() || cycles < 0.0 {
        return None;
    }
    let (n70_eff, z) = decode_theta(params)?;
    if cycles == 0.0 {
        return Some(1.0);
    }
    let loss = (0.3 * (cycles / n70_eff).powf(z)).clamp(0.0, 1.0);
    Some(1.0 - loss)
}

fn validate_and_summarize_samples<P: PointSeries + ?Sized>(
    samples: &P,
) -> Result<(usize, f64, f64), FitError> {
    let mut valid_count = 0usize;
    let mut sum_y = 0.0_f64;
    let mut max_cycles = 0.0_f64;

    for idx in 0..samples.len() {
        let [x, y] = samples.point_at(idx);
        if !x.is_finite() || !y.is_finite() {
            continue;
        }
        if x < 0.0 {
            return Err(FitError::InvalidDomain {
                reason: "cycle count in calibration series must be non-negative",
            });
        }
        if !(0.0..=1.2).contains(&y) {
            return Err(FitError::InvalidDomain {
                reason: "state-of-health capacity fraction must lie within [0.0, 1.2]",
            });
        }
        valid_count += 1;
        sum_y += y;
        max_cycles = max_cycles.max(x);
    }

    if valid_count < 3 {
        return Err(FitError::InsufficientPoints {
            required: 3,
            actual: valid_count,
        });
    }
    if max_cycles <= 0.0 {
        return Err(FitError::InvalidDomain {
            reason: "calibration series must include positive cycle counts",
        });
    }
    Ok((valid_count, sum_y / valid_count as f64, max_cycles))
}

fn total_sum_of_squares<P: PointSeries + ?Sized>(samples: &P, mean_y: f64) -> f64 {
    let mut tss = 0.0_f64;
    for idx in 0..samples.len() {
        let [x, y] = samples.point_at(idx);
        if x.is_finite() && y.is_finite() {
            let diff = y - mean_y;
            tss += diff * diff;
        }
    }
    tss
}

/// Fits $\alpha$ and stretched exponent $z$ of `base` to `(cycles, capacity)` observations at `dod`
/// on the stack via single-pass forward-AD [`levenberg_marquardt_ad`] with zero dynamic heap allocations.
#[verified_engine::verified]
pub fn calibrate_electrochemical_model<P: PointSeries + ?Sized>(
    base: &ElectrochemicalAgingModel,
    samples: &P,
    dod: DepthOfDischarge,
) -> Result<CalibratedAgingFit, FitError> {
    let raw_d = dod.as_f64();
    if !raw_d.is_finite() || raw_d <= 0.0 {
        return Err(FitError::InvalidDomain {
            reason: "depth of discharge must be strictly positive for cycle calibration",
        });
    }
    let (sample_count, mean_y, max_cycles) = validate_and_summarize_samples(samples)?;

    let seed_n70 = base.n70(dod).as_f64().clamp(max_cycles * 0.25, 1.0e7);
    let seed_z = base
        .exponent_z()
        .clamp(MIN_EXPONENT_Z + 0.02, MAX_EXPONENT_Z - 0.02);
    let seed_logit_z = ((seed_z - MIN_EXPONENT_Z) / (MAX_EXPONENT_Z - seed_z)).ln();
    let initial = [seed_n70.max(5.0).ln(), seed_logit_z];

    let theta_full = levenberg_marquardt_ad(samples, &initial, eval_capacity_ad)?;
    let theta_pair = [theta_full[0], theta_full[1]];
    let (fitted_n70, fitted_z) = decode_theta(&theta_pair).ok_or(FitError::InvalidDomain {
        reason: "non-finite calibrated battery aging parameters",
    })?;

    let sse = sum_squared_errors_numeric(samples, &theta_pair, eval_capacity_at).ok_or(
        FitError::InvalidDomain {
            reason: "non-finite calibrated residual sum of squares",
        },
    )?;
    let rmse = (sse / sample_count as f64).sqrt();
    let tss = total_sum_of_squares(samples, mean_y);
    let r_squared = if tss > 1.0e-18 {
        (1.0_f64 - sse / tss).min(1.0)
    } else if sse <= 1.0e-18 {
        1.0
    } else {
        0.0
    };

    let stress = base.arrhenius_factor() * base.c_rate_stress_factor();
    let fitted_alpha = (fitted_n70 * stress) / raw_d.powf(base.beta()).max(1.0e-12);
    let calibrated = base
        .with_dod_law(fitted_alpha, base.beta())
        .with_exponent_z(fitted_z);

    Ok(CalibratedAgingFit {
        model: calibrated,
        fitted_n70_cycles: calibrated.n70(dod).as_f64(),
        fitted_exponent_z: calibrated.exponent_z(),
        rmse,
        r_squared,
        sample_count,
    })
}

impl ElectrochemicalAgingModel {
    /// Calibrates $\alpha$ and stretched exponent $z$ against lab `(cycles, capacity)` points at `dod`
    /// using stack-allocated Levenberg-Marquardt nonlinear least squares.
    #[verified_engine::verified]
    pub fn calibrate_from_cycles<P: PointSeries + ?Sized>(
        &self,
        samples: &P,
        dod: DepthOfDischarge,
    ) -> Result<CalibratedAgingFit, FitError> {
        calibrate_electrochemical_model(self, samples, dod)
    }
}
