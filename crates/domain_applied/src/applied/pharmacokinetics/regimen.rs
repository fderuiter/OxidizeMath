//! Variable-dose multi-event pharmacokinetic superposition.
//!
//! Models clinical and irregular schedules—including loading doses, missed doses, fractional
//! catch-up doses, and mixed immediate/extended-release events—by scaling each event's underlying
//! model through [`PKParameters::with_dose`](super::PKParameters::with_dose) at construction time
//! and evaluating plasma curves with zero heap allocations.

use super::bateman::BatemanModel;
use super::enantiomer::EnantiomerModel;
use super::traits::PharmacokineticModel;
use super::two_pulse::TwoPulseModel;
use crate::error::PharmacokineticsError;

/// Upper bound on scheduled dose administrations in an interactive regimen (`64` events).
pub const MAX_REGIMEN_EVENTS: usize = 64;

/// Release profile applied to an individual [`DoseEvent`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ReleaseFormulation {
    /// Single-pulse absorption governed directly by the underlying model.
    ImmediateRelease,
    /// Biphasic extended-release formulation $C_{\text{XR}}(t) = f_1 C(t) + f_2 C(t - t_{\text{lag}})$.
    ExtendedRelease {
        /// Delay in hours before the delayed release pulse begins ($\ge 0$).
        lag_time_h: f64,
        /// Fraction of the event dose released immediately ($[0, 1]$).
        f1: f64,
        /// Fraction of the event dose released after `lag_time_h` ($[0, 1]$).
        f2: f64,
    },
}

impl ReleaseFormulation {
    /// Validates physical bounds on the formulation parameters.
    #[verified_engine::verified]
    pub fn validate(&self) -> Result<(), PharmacokineticsError> {
        match *self {
            Self::ImmediateRelease => Ok(()),
            Self::ExtendedRelease { lag_time_h, f1, f2 } => {
                if !lag_time_h.is_finite() || lag_time_h < 0.0 {
                    return Err(PharmacokineticsError::InvalidParameter(
                        "Extended-release lag_time_h must be finite and non-negative".to_string(),
                    ));
                }
                if !(0.0..=1.0).contains(&f1) || !(0.0..=1.0).contains(&f2) {
                    return Err(PharmacokineticsError::InvalidParameter(
                        "Extended-release pulse fractions f1 and f2 must lie in [0, 1]".to_string(),
                    ));
                }
                if f1 + f2 > 1.0 + math_commons::registry::TOLERANCE_FAST {
                    return Err(PharmacokineticsError::InvalidParameter(
                        "Extended-release pulse sum f1 + f2 must not exceed 1.0".to_string(),
                    ));
                }
                debug_assert!(lag_time_h >= 0.0);
                debug_assert!(f1 + f2 >= 0.0);
                Ok(())
            }
        }
    }
}

/// Single dose administration event with per-event timing, dose mass, and release formulation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DoseEvent {
    /// Administration time in hours ($\ge 0$).
    pub time_h: f64,
    /// Administered dose mass in milligrams ($\ge 0$).
    pub dose_mg: f64,
    /// Release formulation applied to this event.
    pub formulation: ReleaseFormulation,
}

impl DoseEvent {
    /// Constructs and validates a [`DoseEvent`].
    #[verified_engine::verified]
    pub fn new(
        time_h: f64,
        dose_mg: f64,
        formulation: ReleaseFormulation,
    ) -> Result<Self, PharmacokineticsError> {
        let event = Self {
            time_h,
            dose_mg,
            formulation,
        };
        event.validate()?;
        debug_assert!(event.time_h >= 0.0);
        debug_assert!(event.dose_mg >= 0.0);
        Ok(event)
    }

    /// Constructs and validates an immediate-release [`DoseEvent`].
    #[verified_engine::verified]
    pub fn ir(time_h: f64, dose_mg: f64) -> Result<Self, PharmacokineticsError> {
        Self::new(time_h, dose_mg, ReleaseFormulation::ImmediateRelease)
    }

    /// Constructs and validates a biphasic extended-release [`DoseEvent`].
    #[verified_engine::verified]
    pub fn xr(
        time_h: f64,
        dose_mg: f64,
        lag_time_h: f64,
        f1: f64,
        f2: f64,
    ) -> Result<Self, PharmacokineticsError> {
        Self::new(
            time_h,
            dose_mg,
            ReleaseFormulation::ExtendedRelease { lag_time_h, f1, f2 },
        )
    }

    /// Validates that `time_h`, `dose_mg`, and `formulation` satisfy physical invariants.
    #[verified_engine::verified]
    pub fn validate(&self) -> Result<(), PharmacokineticsError> {
        if !self.time_h.is_finite() || self.time_h < 0.0 {
            return Err(PharmacokineticsError::InvalidParameter(
                "DoseEvent time_h must be finite and non-negative".to_string(),
            ));
        }
        if !self.dose_mg.is_finite() || self.dose_mg < 0.0 {
            return Err(PharmacokineticsError::InvalidParameter(
                "DoseEvent dose_mg must be finite and non-negative".to_string(),
            ));
        }
        self.formulation.validate()?;
        debug_assert!(self.time_h >= 0.0);
        debug_assert!(self.dose_mg >= 0.0);
        Ok(())
    }
}

/// Trait implemented by linear pharmacokinetic models whose dose mass can be scaled per event.
pub trait RegimenModel: PharmacokineticModel + Copy {
    /// Returns a copy of `self` scaled to `dose_mg` via `PKParameters::with_dose`.
    fn with_event_dose(&self, dose_mg: f64) -> Result<Self, PharmacokineticsError>;
}

impl RegimenModel for BatemanModel {
    #[verified_engine::verified]
    fn with_event_dose(&self, dose_mg: f64) -> Result<Self, PharmacokineticsError> {
        self.with_dose(dose_mg)
    }
}

impl RegimenModel for EnantiomerModel {
    #[verified_engine::verified]
    fn with_event_dose(&self, dose_mg: f64) -> Result<Self, PharmacokineticsError> {
        self.with_dose(dose_mg)
    }
}

impl<M: RegimenModel> RegimenModel for TwoPulseModel<M> {
    #[verified_engine::verified]
    fn with_event_dose(&self, dose_mg: f64) -> Result<Self, PharmacokineticsError> {
        if !self.lag_time.is_finite() || self.lag_time < 0.0 {
            return Err(PharmacokineticsError::InvalidParameter(
                "TwoPulseModel lag_time must be finite and non-negative".to_string(),
            ));
        }
        if !(0.0..=1.0).contains(&self.f1) || !(0.0..=1.0).contains(&self.f2) {
            return Err(PharmacokineticsError::InvalidParameter(
                "TwoPulseModel pulse fractions f1 and f2 must lie in [0, 1]".to_string(),
            ));
        }
        self.with_dose(dose_mg)
    }
}

/// Evaluates a single pre-scaled event contribution at elapsed time `elapsed_h`.
#[verified_engine::verified]
fn evaluate_event_model<M: PharmacokineticModel>(
    model: &M,
    formulation: ReleaseFormulation,
    elapsed_h: f64,
) -> f64 {
    if elapsed_h < 0.0 {
        return 0.0;
    }
    let out = match formulation {
        ReleaseFormulation::ImmediateRelease => model.concentration(elapsed_h),
        ReleaseFormulation::ExtendedRelease { lag_time_h, f1, f2 } => {
            let c1 = model.concentration(elapsed_h);
            let c2 = if elapsed_h >= lag_time_h {
                model.concentration(elapsed_h - lag_time_h)
            } else {
                0.0
            };
            f1 * c1 + f2 * c2
        }
    };
    debug_assert!(out >= 0.0 || out.is_nan());
    debug_assert!(elapsed_h >= 0.0);
    out
}

/// Analytical multi-dose superposition supporting event-specific timing, dose mass, and release.
///
/// Pre-scales per-event models into a fixed-capacity stack array (`<= MAX_REGIMEN_EVENTS`) during
/// [`RegimenSuperposition::new`] so concentration evaluation and slice sampling perform **zero heap
/// allocations** inside locked `VerifiedAllocator` regions.
#[derive(Debug, Clone, Copy)]
pub struct RegimenSuperposition<M: Copy> {
    base_model: M,
    events: [DoseEvent; MAX_REGIMEN_EVENTS],
    scaled_models: [M; MAX_REGIMEN_EVENTS],
    len: usize,
}

impl<M: RegimenModel> RegimenSuperposition<M> {
    /// Creates a validated variable-dose regimen from `base_model` and up to `64` [`DoseEvent`]s.
    #[verified_engine::verified]
    pub fn new(base_model: M, events: &[DoseEvent]) -> Result<Self, PharmacokineticsError> {
        if events.len() > MAX_REGIMEN_EVENTS {
            return Err(PharmacokineticsError::InvalidParameter(format!(
                "Regimen schedule length {} exceeds MAX_REGIMEN_EVENTS ({})",
                events.len(),
                MAX_REGIMEN_EVENTS
            )));
        }
        let placeholder_event = DoseEvent {
            time_h: 0.0,
            dose_mg: 0.0,
            formulation: ReleaseFormulation::ImmediateRelease,
        };
        let mut stored_events = [placeholder_event; MAX_REGIMEN_EVENTS];
        let mut scaled_models = [base_model; MAX_REGIMEN_EVENTS];

        let mut idx = 0;
        while idx < events.len() {
            let ev = events[idx];
            ev.validate()?;
            stored_events[idx] = ev;
            scaled_models[idx] = base_model.with_event_dose(ev.dose_mg)?;
            idx += 1;
        }
        debug_assert!(idx <= MAX_REGIMEN_EVENTS);
        debug_assert_eq!(idx, events.len());

        Ok(Self {
            base_model,
            events: stored_events,
            scaled_models,
            len: events.len(),
        })
    }

    /// Returns a reference to the unscaled base model.
    #[verified_engine::verified]
    pub fn base_model(&self) -> &M {
        debug_assert!(self.len <= MAX_REGIMEN_EVENTS);
        &self.base_model
    }

    /// Returns the active slice of scheduled [`DoseEvent`]s.
    #[verified_engine::verified]
    pub fn events(&self) -> &[DoseEvent] {
        debug_assert!(self.len <= MAX_REGIMEN_EVENTS);
        &self.events[..self.len]
    }

    /// Returns the number of scheduled [`DoseEvent`]s.
    #[verified_engine::verified]
    pub fn len(&self) -> usize {
        debug_assert!(self.len <= MAX_REGIMEN_EVENTS);
        self.len
    }

    /// Returns `true` if no [`DoseEvent`]s are scheduled.
    #[verified_engine::verified]
    pub fn is_empty(&self) -> bool {
        debug_assert!(self.len <= MAX_REGIMEN_EVENTS);
        self.len == 0
    }

    /// Samples total plasma concentrations at `times` into caller buffer `out` with zero allocations.
    #[verified_engine::verified]
    pub fn sample_into(&self, times: &[f64], out: &mut [f64]) -> Result<(), PharmacokineticsError> {
        if times.len() != out.len() {
            return Err(PharmacokineticsError::InvalidParameter(
                "sample_into requires times and out slices to have equal length".to_string(),
            ));
        }
        let mut idx = 0;
        while idx < times.len() {
            out[idx] = self.concentration(times[idx]);
            idx += 1;
        }
        debug_assert_eq!(idx, out.len());
        debug_assert!(self.len <= MAX_REGIMEN_EVENTS);
        Ok(())
    }
}

impl<M: RegimenModel> PharmacokineticModel for RegimenSuperposition<M> {
    #[verified_engine::verified]
    fn concentration(&self, t: f64) -> f64 {
        let mut total = 0.0;
        let mut idx = 0;
        while idx < self.len {
            let ev = self.events[idx];
            if t >= ev.time_h {
                total +=
                    evaluate_event_model(&self.scaled_models[idx], ev.formulation, t - ev.time_h);
            }
            idx += 1;
        }
        debug_assert!(total >= 0.0 || total.is_nan());
        debug_assert!(self.len <= MAX_REGIMEN_EVENTS);
        total
    }
}

impl RegimenSuperposition<EnantiomerModel> {
    /// Evaluates the superposed d-enantiomer plasma concentration at time `t` (hours).
    #[verified_engine::verified]
    pub fn d_concentration(&self, t: f64) -> f64 {
        let mut total = 0.0;
        let mut idx = 0;
        while idx < self.len {
            let ev = self.events[idx];
            let dt = t - ev.time_h;
            if dt >= 0.0 {
                total += match ev.formulation {
                    ReleaseFormulation::ImmediateRelease => {
                        self.scaled_models[idx].d_concentration(dt)
                    }
                    ReleaseFormulation::ExtendedRelease { lag_time_h, f1, f2 } => {
                        let c1 = self.scaled_models[idx].d_concentration(dt);
                        let c2 = if dt >= lag_time_h {
                            self.scaled_models[idx].d_concentration(dt - lag_time_h)
                        } else {
                            0.0
                        };
                        f1 * c1 + f2 * c2
                    }
                };
            }
            idx += 1;
        }
        debug_assert!(total >= 0.0 || total.is_nan());
        total
    }

    /// Evaluates the superposed l-enantiomer plasma concentration at time `t` (hours).
    #[verified_engine::verified]
    pub fn l_concentration(&self, t: f64) -> f64 {
        let mut total = 0.0;
        let mut idx = 0;
        while idx < self.len {
            let ev = self.events[idx];
            let dt = t - ev.time_h;
            if dt >= 0.0 {
                total += match ev.formulation {
                    ReleaseFormulation::ImmediateRelease => {
                        self.scaled_models[idx].l_concentration(dt)
                    }
                    ReleaseFormulation::ExtendedRelease { lag_time_h, f1, f2 } => {
                        let c1 = self.scaled_models[idx].l_concentration(dt);
                        let c2 = if dt >= lag_time_h {
                            self.scaled_models[idx].l_concentration(dt - lag_time_h)
                        } else {
                            0.0
                        };
                        f1 * c1 + f2 * c2
                    }
                };
            }
            idx += 1;
        }
        debug_assert!(total >= 0.0 || total.is_nan());
        total
    }

    /// Samples d-isomer, l-isomer, and total concentrations into caller slices with zero allocations.
    #[verified_engine::verified]
    pub fn sample_isomers_into(
        &self,
        times: &[f64],
        d_out: &mut [f64],
        l_out: &mut [f64],
        total_out: &mut [f64],
    ) -> Result<(), PharmacokineticsError> {
        if times.len() != d_out.len()
            || times.len() != l_out.len()
            || times.len() != total_out.len()
        {
            return Err(PharmacokineticsError::InvalidParameter(
                "sample_isomers_into requires equal slice lengths".to_string(),
            ));
        }
        let mut idx = 0;
        while idx < times.len() {
            let d_val = self.d_concentration(times[idx]);
            let l_val = self.l_concentration(times[idx]);
            d_out[idx] = d_val;
            l_out[idx] = l_val;
            total_out[idx] = d_val + l_val;
            idx += 1;
        }
        debug_assert_eq!(idx, times.len());
        Ok(())
    }
}
