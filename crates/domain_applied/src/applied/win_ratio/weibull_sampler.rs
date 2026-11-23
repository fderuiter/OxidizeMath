//! Deterministic two-step inverse-CDF sampling for [`super::simulation::BivariateWeibullModel`].

use crate::applied::clinical_trials::types::{ClinicalTrialError, SurvivalTime};
use oxidize_core::rng::OxidizeRng;
use rand::RngCore;

use super::censoring::CensoredOutcome;
use super::simulation::{BivariateWeibullModel, SimulationParams};

const MAX_NEWTON_ITERS: usize = 16;
const NEWTON_TOLERANCE: f64 = 1e-13;

/// A simulated two-tier composite patient record `[Fatal, Non-Fatal]` after follow-up censoring.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SimulatedSubject {
    /// Tier 1 prioritized fatal endpoint (e.g., cardiovascular death).
    pub fatal: CensoredOutcome,
    /// Tier 2 prioritized non-fatal endpoint (e.g., heart-failure hospitalization).
    pub nonfatal: CensoredOutcome,
}

impl Default for SimulatedSubject {
    fn default() -> Self {
        let zero = SurvivalTime::new(0.0).expect("0.0 is valid SurvivalTime");
        Self {
            fatal: CensoredOutcome {
                time: zero,
                event_occurred: false,
            },
            nonfatal: CensoredOutcome {
                time: zero,
                event_occurred: false,
            },
        }
    }
}

#[inline]
pub(crate) fn open_unit_uniform(rng: &mut OxidizeRng) -> f64 {
    let bits = rng.next_u64() >> 11;
    let u = (bits as f64 + 0.5) * (1.0 / ((1_u64 << 53) as f64));
    debug_assert!(u > 0.0 && u < 1.0);
    u
}

#[inline]
fn solve_conditional_weibull_increment(w0: f64, v: f64, beta: f64) -> f64 {
    debug_assert!(w0 > 0.0 && v > 0.0 && beta >= 0.0);
    if beta <= 1e-15 {
        return v;
    }
    let mut s = (v / (1.0 + beta / (w0 + v))).clamp(0.0, v);
    for _ in 0..MAX_NEWTON_ITERS {
        let f_val = s + beta * (s / w0).ln_1p() - v;
        if f_val.abs() <= NEWTON_TOLERANCE {
            break;
        }
        let f_deriv = 1.0 + beta / (w0 + s);
        s = (s - f_val / f_deriv).clamp(0.0, v);
    }
    s
}

impl BivariateWeibullModel {
    /// Validates that scale parameters `lambda1`, `lambda2` are positive finite and
    /// association shape `alpha` lies in `(0.0, 1.0]`.
    #[verified_engine::verified]
    pub fn validate(&self) -> Result<(), ClinicalTrialError> {
        if !self.lambda1.is_finite() || self.lambda1 <= 0.0 {
            return Err(ClinicalTrialError::InvalidData(format!(
                "lambda1 must be positive and finite, got {}",
                self.lambda1
            )));
        }
        if !self.lambda2.is_finite() || self.lambda2 <= 0.0 {
            return Err(ClinicalTrialError::InvalidData(format!(
                "lambda2 must be positive and finite, got {}",
                self.lambda2
            )));
        }
        if !self.alpha.is_finite() || self.alpha <= 0.0 || self.alpha > 1.0 {
            return Err(ClinicalTrialError::InvalidData(format!(
                "alpha must be finite in (0, 1], got {}",
                self.alpha
            )));
        }
        Ok(())
    }

    /// Samples an uncensored `(fatal_time, nonfatal_time)` pair via exact two-step inverse-CDF
    /// transformation driven deterministically by [`OxidizeRng`].
    ///
    /// # Errors
    /// Returns [`ClinicalTrialError::InvalidData`] if model parameters fail [`Self::validate`].
    #[verified_engine::verified]
    pub fn try_sample_event_times(
        &self,
        rng: &mut OxidizeRng,
    ) -> Result<(f64, f64), ClinicalTrialError> {
        self.validate()?;
        let u1 = open_unit_uniform(rng);
        let u2 = open_unit_uniform(rng);
        let w0 = -u1.ln();
        let inv_alpha = 1.0 / self.alpha;
        let z0 = w0.powf(inv_alpha);
        let t = z0 / self.lambda1;

        let v = -u2.ln();
        let s = solve_conditional_weibull_increment(w0, v, inv_alpha - 1.0);
        let z = (w0 + s).powf(inv_alpha);
        let x = ((z - z0).max(0.0)) / self.lambda2;
        debug_assert!(t.is_finite() && t >= 0.0 && x.is_finite() && x >= 0.0);
        Ok((t, x))
    }

    /// Samples a right-censored [`SimulatedSubject`] under administrative horizon `horizon`
    /// and optional exponential random loss-to-follow-up rate `dropout_rate >= 0.0`.
    ///
    /// # Errors
    /// Returns [`ClinicalTrialError::InvalidData`] if model or censoring parameters are invalid.
    #[verified_engine::verified]
    pub fn try_sample_censored_subject(
        &self,
        rng: &mut OxidizeRng,
        horizon: f64,
        dropout_rate: f64,
    ) -> Result<SimulatedSubject, ClinicalTrialError> {
        if !horizon.is_finite() || horizon <= 0.0 {
            return Err(ClinicalTrialError::InvalidData(format!(
                "Censoring horizon must be positive and finite, got {horizon}"
            )));
        }
        if !dropout_rate.is_finite() || dropout_rate < 0.0 {
            return Err(ClinicalTrialError::InvalidData(format!(
                "Dropout rate must be non-negative and finite, got {dropout_rate}"
            )));
        }
        let (raw_t, raw_x) = self.try_sample_event_times(rng)?;
        let follow_up = if dropout_rate > 0.0 {
            horizon.min(-open_unit_uniform(rng).ln() / dropout_rate)
        } else {
            horizon
        };
        let fatal_obs = raw_t.min(follow_up);
        let fatal_event = raw_t <= follow_up;
        let nonfatal_cap = follow_up.min(raw_t);
        let nonfatal_obs = raw_x.min(nonfatal_cap);
        let nonfatal_event = raw_x <= nonfatal_cap;

        Ok(SimulatedSubject {
            fatal: CensoredOutcome::new(fatal_obs, fatal_event)?,
            nonfatal: CensoredOutcome::new(nonfatal_obs, nonfatal_event)?,
        })
    }
}

impl SimulationParams {
    /// Validates all four [`SimulationParams`] scalars (`lambda1 > 0`, `lambda2 > 0`,
    /// `0 < alpha <= 1`, and `theta > 0`).
    #[verified_engine::verified]
    pub fn validate(&self) -> Result<(), ClinicalTrialError> {
        self.control_model().validate()?;
        if !self.theta.is_finite() || self.theta <= 0.0 {
            return Err(ClinicalTrialError::InvalidData(format!(
                "theta must be positive and finite, got {}",
                self.theta
            )));
        }
        self.treatment_model().validate()
    }
}
