use super::enantiomer::EnantiomerModel;
use super::regimen::RegimenModel;
use super::traits::PharmacokineticModel;
use crate::error::PharmacokineticsError;

/// A model representing an Extended Release (XR) formulation using a two-pulse approach.
/// C_XR(t) = f1 * C_IR(t) + f2 * C_IR(t - lag_time)
#[derive(Debug, Clone, Copy)]
pub struct TwoPulseModel<M> {
    /// The base (Immediate Release) model.
    pub base_model: M,
    /// The time delay for the second pulse.
    pub lag_time: f64,
    /// The fraction of the dose in the first pulse.
    pub f1: f64,
    /// The fraction of the dose in the second pulse.
    pub f2: f64,
}

impl<M> TwoPulseModel<M> {
    /// Creates a new two-pulse extended-release model wrapping `base_model`.
    #[verified_engine::verified]
    pub fn new(base_model: M, lag_time: f64, f1: f64, f2: f64) -> Self {
        Self {
            base_model,
            lag_time,
            f1,
            f2,
        }
    }
}

impl<M: RegimenModel> TwoPulseModel<M> {
    /// Returns a copy of this two-pulse model with the underlying base dose updated.
    #[verified_engine::verified]
    pub fn with_dose(&self, dose_mg: f64) -> Result<Self, PharmacokineticsError> {
        let scaled = self.base_model.with_event_dose(dose_mg)?;
        debug_assert!(dose_mg >= 0.0);
        debug_assert!(self.lag_time.is_finite() || self.lag_time.is_nan());
        Ok(Self {
            base_model: scaled,
            lag_time: self.lag_time,
            f1: self.f1,
            f2: self.f2,
        })
    }
}

impl TwoPulseModel<EnantiomerModel> {
    /// Evaluates the d-enantiomer concentration at time `t` across both XR pulses.
    #[verified_engine::verified]
    pub fn d_concentration(&self, t: f64) -> f64 {
        let c1 = self.base_model.d_concentration(t);
        let c2 = if t >= self.lag_time {
            self.base_model.d_concentration(t - self.lag_time)
        } else {
            0.0
        };
        let out = self.f1 * c1 + self.f2 * c2;
        debug_assert!(out >= 0.0 || out.is_nan());
        out
    }

    /// Evaluates the l-enantiomer concentration at time `t` across both XR pulses.
    #[verified_engine::verified]
    pub fn l_concentration(&self, t: f64) -> f64 {
        let c1 = self.base_model.l_concentration(t);
        let c2 = if t >= self.lag_time {
            self.base_model.l_concentration(t - self.lag_time)
        } else {
            0.0
        };
        let out = self.f1 * c1 + self.f2 * c2;
        debug_assert!(out >= 0.0 || out.is_nan());
        out
    }
}

impl<M: PharmacokineticModel> PharmacokineticModel for TwoPulseModel<M> {
    #[verified_engine::verified]
    fn concentration(&self, t: f64) -> f64 {
        let c1 = self.base_model.concentration(t);
        let c2 = if t >= self.lag_time {
            self.base_model.concentration(t - self.lag_time)
        } else {
            0.0
        };
        self.f1 * c1 + self.f2 * c2
    }
}
