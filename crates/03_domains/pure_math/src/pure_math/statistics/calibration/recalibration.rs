//! Parametric and non-parametric post-hoc probability recalibrators:
//! damped Newton-Raphson [`PlattScaler`], convex [`TemperatureScaler`], and `PAVA`
//! [`IsotonicCalibrator`].

use math_commons::error::{MathError, MathResult};
use math_commons::math_kernel::types::Dimension;
use math_commons::primitives::{PositiveFloat, UnitInterval};
use serde::{Deserialize, Serialize};

pub use super::isotonic::{
    IsotonicCalibrator, PavaBlock, fit_pava_sorted_into, predict_pava_blocks,
};

/// Solver hyperparameters for damped Newton-Raphson [`PlattScaler`] fitting.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PlattConfig {
    /// Maximum damped Newton-Raphson iterations (`>= 1`).
    pub max_iter: usize,
    /// Gradient and step Euclidean norm convergence tolerance (`> 0.0`).
    pub tol: f64,
    /// Non-negative $L_2$ ridge penalty preventing divergence on linearly separable logits.
    pub l2_reg: f64,
    /// Whether to apply Platt (1999) Bayesian target smoothing $(N_+ + 1)/(N_+ + 2)$ and $1/(N_- + 2)$.
    pub target_smoothing: bool,
}

impl Default for PlattConfig {
    fn default() -> Self {
        Self {
            max_iter: 100,
            tol: 1e-9,
            l2_reg: 1e-6,
            target_smoothing: false,
        }
    }
}

#[doc = include_str!("../../../../../../../docs/math_specs/snippets/platt_scaler.md")]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PlattScaler {
    /// Learned multiplicative logit slope $a$.
    pub scale: f64,
    /// Learned additive logit intercept $b$.
    pub bias: f64,
}

fn stable_sigmoid(eta: f64) -> f64 {
    if eta >= 0.0 {
        1.0 / (1.0 + (-eta).exp())
    } else {
        let e = eta.exp();
        e / (1.0 + e)
    }
}

fn softplus_bce(eta: f64, target: f64) -> f64 {
    eta.max(0.0) - eta * target + (-eta.abs()).exp().ln_1p()
}

fn validate_logit_outcomes(logits: &[f64], outcomes: &[bool]) -> MathResult<(usize, usize)> {
    if logits.len() != outcomes.len() {
        return Err(MathError::DimensionMismatch {
            expected: Dimension(outcomes.len()),
            actual: Dimension(logits.len()),
        });
    }
    if logits.is_empty() {
        return Err(MathError::NumericalError {
            reason: "Calibration logits must be non-empty",
        });
    }
    if logits.iter().any(|z| !z.is_finite()) {
        return Err(MathError::NumericalError {
            reason: "Calibration logits must be finite",
        });
    }
    let n_pos = outcomes.iter().filter(|&&y| y).count();
    let n_neg = outcomes.len() - n_pos;
    if n_pos == 0 || n_neg == 0 {
        return Err(MathError::NumericalError {
            reason: "Calibration fitting requires both positive and negative outcomes",
        });
    }
    Ok((n_pos, n_neg))
}

fn platt_stats(
    logits: &[f64],
    outcomes: &[bool],
    scale: f64,
    bias: f64,
    targets: (f64, f64),
    l2_reg: f64,
) -> (f64, [f64; 2], [f64; 3]) {
    let inv_n = 1.0 / (logits.len() as f64);
    let (mut loss, mut g_a, mut g_b) = (0.0, 0.0, 0.0);
    let (mut h_aa, mut h_ab, mut h_bb) = (0.0, 0.0, 0.0);
    for (&z, &y) in logits.iter().zip(outcomes.iter()) {
        let t = if y { targets.0 } else { targets.1 };
        let eta = scale * z + bias;
        let p = stable_sigmoid(eta);
        let diff = p - t;
        let w = (p * (1.0 - p)).max(1e-12);
        loss += softplus_bce(eta, t);
        g_a += diff * z;
        g_b += diff;
        h_aa += w * z * z;
        h_ab += w * z;
        h_bb += w;
    }
    let reg_loss = 0.5 * l2_reg * (scale * scale + bias * bias);
    (
        loss * inv_n + reg_loss,
        [g_a * inv_n + l2_reg * scale, g_b * inv_n + l2_reg * bias],
        [h_aa * inv_n + l2_reg, h_ab * inv_n, h_bb * inv_n + l2_reg],
    )
}

fn solve_damped_2x2(hess: [f64; 3], grad: [f64; 2], damping: f64) -> Option<[f64; 2]> {
    let a11 = hess[0] + damping;
    let a12 = hess[1];
    let a22 = hess[2] + damping;
    let det = a11 * a22 - a12 * a12;
    if !det.is_finite() || det <= 1e-18 {
        return None;
    }
    let inv_det = 1.0 / det;
    Some([
        -(a22 * grad[0] - a12 * grad[1]) * inv_det,
        -(-a12 * grad[0] + a11 * grad[1]) * inv_det,
    ])
}

impl PlattScaler {
    /// Fits logistic Platt scaling on held-out `(logits, outcomes)` using default solver settings.
    #[verified_engine::verified]
    pub fn fit(logits: &[f64], outcomes: &[bool]) -> MathResult<Self> {
        Self::fit_with_config(logits, outcomes, PlattConfig::default())
    }

    /// Fits logistic Platt scaling via zero-allocation damped Newton-Raphson with backtracking.
    #[verified_engine::verified]
    pub fn fit_with_config(
        logits: &[f64],
        outcomes: &[bool],
        config: PlattConfig,
    ) -> MathResult<Self> {
        let (n_pos, n_neg) = validate_logit_outcomes(logits, outcomes)?;
        if config.max_iter == 0
            || !config.tol.is_finite()
            || config.tol <= 0.0
            || !config.l2_reg.is_finite()
            || config.l2_reg < 0.0
        {
            return Err(MathError::NumericalError {
                reason: "Invalid PlattConfig hyperparameters",
            });
        }
        let targets = if config.target_smoothing {
            (
                (n_pos as f64 + 1.0) / (n_pos as f64 + 2.0),
                1.0 / (n_neg as f64 + 2.0),
            )
        } else {
            (1.0, 0.0)
        };
        let (mut scale, mut bias, mut damping) = (1.0, 0.0, 1e-6);
        for _ in 0..config.max_iter {
            let (loss, grad, hess) =
                platt_stats(logits, outcomes, scale, bias, targets, config.l2_reg);
            if grad[0].hypot(grad[1]) <= config.tol {
                break;
            }
            let Some(step) = solve_damped_2x2(hess, grad, damping) else {
                damping *= 10.0;
                continue;
            };
            let mut alpha = 1.0;
            let mut accepted = false;
            for _ in 0..24 {
                let (cand_a, cand_b) = (scale + alpha * step[0], bias + alpha * step[1]);
                let (cand_loss, _, _) =
                    platt_stats(logits, outcomes, cand_a, cand_b, targets, config.l2_reg);
                if cand_loss <= loss {
                    scale = cand_a;
                    bias = cand_b;
                    accepted = true;
                    break;
                }
                alpha *= 0.5;
            }
            if !accepted {
                damping *= 10.0;
                if damping > 1e8 {
                    break;
                }
                continue;
            }
            damping = if alpha == 1.0 {
                (damping * 0.5).max(1e-10)
            } else {
                (damping * 2.0).min(1e8)
            };
            if (alpha * step[0]).hypot(alpha * step[1]) <= config.tol {
                break;
            }
        }
        Ok(Self { scale, bias })
    }

    /// Maps a scalar `logit` to a calibrated [`UnitInterval`] probability.
    #[verified_engine::verified]
    pub fn predict_proba(&self, logit: f64) -> MathResult<UnitInterval> {
        if !logit.is_finite() {
            return Err(MathError::NumericalError {
                reason: "Input logit must be finite",
            });
        }
        let p = stable_sigmoid(self.scale * logit + self.bias).clamp(0.0, 1.0);
        UnitInterval::new(p)
    }

    /// Evaluates calibrated probabilities for `logits` into `out` with zero heap allocations.
    #[verified_engine::verified]
    pub fn predict_into(&self, logits: &[f64], out: &mut [UnitInterval]) -> MathResult<()> {
        if logits.len() != out.len() {
            return Err(MathError::DimensionMismatch {
                expected: Dimension(logits.len()),
                actual: Dimension(out.len()),
            });
        }
        for (&z, dst) in logits.iter().zip(out.iter_mut()) {
            *dst = self.predict_proba(z)?;
        }
        Ok(())
    }
}

#[doc = include_str!("../../../../../../../docs/math_specs/snippets/temperature_scaler.md")]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TemperatureScaler {
    /// Strictly positive temperature parameter $T > 0$ minimizing held-out `log_loss`.
    pub temperature: PositiveFloat,
}

fn binary_beta_stats(logits: &[f64], outcomes: &[bool], beta: f64) -> (f64, f64, f64) {
    let inv_n = 1.0 / (logits.len() as f64);
    let (mut loss, mut grad, mut hess) = (0.0, 0.0, 0.0);
    for (&z, &y) in logits.iter().zip(outcomes.iter()) {
        let t = if y { 1.0 } else { 0.0 };
        let eta = beta * z;
        let p = stable_sigmoid(eta);
        loss += softplus_bce(eta, t);
        grad += (p - t) * z;
        hess += p * (1.0 - p) * z * z;
    }
    (loss * inv_n, grad * inv_n, hess * inv_n)
}

fn multiclass_beta_stats(
    logits_flat: &[f64],
    num_classes: usize,
    targets: &[usize],
    beta: f64,
) -> (f64, f64, f64) {
    let inv_n = 1.0 / (targets.len() as f64);
    let (mut loss, mut grad, mut hess) = (0.0, 0.0, 0.0);
    for (row_idx, &target) in targets.iter().enumerate() {
        let row = &logits_flat[row_idx * num_classes..(row_idx + 1) * num_classes];
        let mut max_eta = f64::NEG_INFINITY;
        for &z in row {
            max_eta = max_eta.max(beta * z);
        }
        let (mut sum_exp, mut weighted_z, mut weighted_z2) = (0.0, 0.0, 0.0);
        for &z in row {
            let w = (beta * z - max_eta).exp();
            sum_exp += w;
            weighted_z += w * z;
            weighted_z2 += w * z * z;
        }
        let log_z = max_eta + sum_exp.ln();
        let mean_z = weighted_z / sum_exp;
        let var_z = (weighted_z2 / sum_exp - mean_z * mean_z).max(0.0);
        loss += log_z - beta * row[target];
        grad += mean_z - row[target];
        hess += var_z;
    }
    (loss * inv_n, grad * inv_n, hess * inv_n)
}

fn solve_convex_beta<F>(eval: F) -> MathResult<PositiveFloat>
where
    F: Fn(f64) -> (f64, f64, f64),
{
    let (min_beta, max_beta) = (1e-3_f64, 1e3_f64);
    let mut beta = 1.0_f64;
    let mut damping = 1e-8_f64;
    for _ in 0..80 {
        let (loss, grad, hess) = eval(beta);
        if grad.abs() <= 1e-10 {
            break;
        }
        let denom = (hess + damping).max(1e-12);
        let raw_step = -grad / denom;
        let mut alpha = 1.0_f64;
        let mut accepted = false;
        for _ in 0..24 {
            let cand = (beta + alpha * raw_step).clamp(min_beta, max_beta);
            let (cand_loss, _, _) = eval(cand);
            if cand_loss <= loss {
                let delta = (cand - beta).abs();
                beta = cand;
                accepted = true;
                if delta <= 1e-10 {
                    return PositiveFloat::new(1.0 / beta);
                }
                break;
            }
            alpha *= 0.5;
        }
        if !accepted {
            damping *= 10.0;
            if damping > 1e8 {
                break;
            }
        } else {
            damping = (damping * 0.5).max(1e-10);
        }
    }
    PositiveFloat::new(1.0 / beta)
}

impl TemperatureScaler {
    /// Constructs a [`TemperatureScaler`] from an explicit positive `temperature`.
    #[verified_engine::verified]
    pub fn new(temperature: PositiveFloat) -> Self {
        Self { temperature }
    }

    /// Fits $T > 0$ minimizing binary held-out `log_loss` via convex inverse-temperature Newton steps.
    #[verified_engine::verified]
    pub fn fit_binary(logits: &[f64], outcomes: &[bool]) -> MathResult<Self> {
        let _ = validate_logit_outcomes(logits, outcomes)?;
        let temperature = solve_convex_beta(|beta| binary_beta_stats(logits, outcomes, beta))?;
        Ok(Self { temperature })
    }

    /// Fits $T > 0$ minimizing multiclass held-out `log_loss` over row-major `logits_flat`.
    #[verified_engine::verified]
    pub fn fit_multiclass(
        logits_flat: &[f64],
        num_classes: usize,
        targets: &[usize],
    ) -> MathResult<Self> {
        if num_classes < 2 || targets.is_empty() {
            return Err(MathError::NumericalError {
                reason: "fit_multiclass requires num_classes >= 2 and non-empty targets",
            });
        }
        let expected = targets
            .len()
            .checked_mul(num_classes)
            .ok_or(MathError::NumericalError {
                reason: "targets.len() * num_classes overflowed usize",
            })?;
        if logits_flat.len() != expected {
            return Err(MathError::DimensionMismatch {
                expected: Dimension(expected),
                actual: Dimension(logits_flat.len()),
            });
        }
        if logits_flat.iter().any(|z| !z.is_finite()) || targets.iter().any(|&t| t >= num_classes) {
            return Err(MathError::NumericalError {
                reason: "fit_multiclass requires finite logits and in-bounds class targets",
            });
        }
        let temperature = solve_convex_beta(|beta| {
            multiclass_beta_stats(logits_flat, num_classes, targets, beta)
        })?;
        Ok(Self { temperature })
    }

    /// Predicts calibrated binary probability $\sigma(z / T)$ as a [`UnitInterval`].
    #[verified_engine::verified]
    pub fn predict_binary(&self, logit: f64) -> MathResult<UnitInterval> {
        if !logit.is_finite() {
            return Err(MathError::NumericalError {
                reason: "Input logit must be finite",
            });
        }
        let p = stable_sigmoid(logit / self.temperature.value()).clamp(0.0, 1.0);
        UnitInterval::new(p)
    }

    /// Evaluates calibrated softmax probabilities $\text{softmax}(z / T)$ into `out_probs` with zero allocations.
    #[verified_engine::verified]
    pub fn predict_multiclass_into(
        &self,
        logits: &[f64],
        out_probs: &mut [UnitInterval],
    ) -> MathResult<()> {
        if logits.len() != out_probs.len() {
            return Err(MathError::DimensionMismatch {
                expected: Dimension(logits.len()),
                actual: Dimension(out_probs.len()),
            });
        }
        if logits.is_empty() || logits.iter().any(|z| !z.is_finite()) {
            return Err(MathError::NumericalError {
                reason: "predict_multiclass_into requires non-empty finite logits",
            });
        }
        let inv_t = 1.0 / self.temperature.value();
        let mut max_scaled = f64::NEG_INFINITY;
        for &z in logits {
            max_scaled = max_scaled.max(z * inv_t);
        }
        let mut sum = 0.0;
        for &z in logits {
            sum += (z * inv_t - max_scaled).exp();
        }
        for (&z, dst) in logits.iter().zip(out_probs.iter_mut()) {
            let p = ((z * inv_t - max_scaled).exp() / sum).clamp(0.0, 1.0);
            *dst = UnitInterval::new(p)?;
        }
        Ok(())
    }
}
