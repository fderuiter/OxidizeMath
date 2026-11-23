//! Right-censored time-to-event outcomes, Pocock/Finkelstein-Schoenfeld comparators,
//! and per-tier hierarchical counters for composite win-ratio analyses.

use crate::applied::clinical_trials::survival_analysis::Observation;
use crate::applied::clinical_trials::types::{ClinicalTrialError, SurvivalTime};

use super::pair_comparison::{ComparisonResult, OutcomeComparator};

/// A right-censored time-to-event outcome for hierarchical composite win-ratio comparisons.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CensoredOutcome {
    /// Observed event time or right-censoring follow-up time (`>= 0.0` and finite).
    pub time: SurvivalTime,
    /// `true` when the adverse event occurred at `time`; `false` when right-censored.
    pub event_occurred: bool,
}

impl CensoredOutcome {
    /// Constructs a validated [`CensoredOutcome`] from a raw `f64` follow-up time and event indicator.
    ///
    /// # Errors
    /// Returns [`ClinicalTrialError::InvalidData`] if `time` is negative or non-finite.
    #[verified_engine::verified]
    pub fn new(time: f64, event_occurred: bool) -> Result<Self, ClinicalTrialError> {
        let validated = SurvivalTime::new(time)?;
        debug_assert!(validated.as_f64().is_finite() && validated.as_f64() >= 0.0);
        Ok(Self {
            time: validated,
            event_occurred,
        })
    }

    /// Applies an administrative follow-up horizon `c`, right-censoring events after `c`.
    #[verified_engine::verified]
    #[must_use]
    pub fn censored_at(&self, horizon: Option<f64>) -> (f64, bool) {
        let raw_t = self.time.as_f64();
        debug_assert!(raw_t.is_finite() && raw_t >= 0.0);
        if let Some(c) = horizon
            && raw_t > c
        {
            return (c, false);
        }
        (raw_t, self.event_occurred)
    }
}

impl From<Observation> for CensoredOutcome {
    fn from(obs: Observation) -> Self {
        Self {
            time: obs.time,
            event_occurred: obs.event_occurred,
        }
    }
}

impl From<CensoredOutcome> for Observation {
    fn from(outcome: CensoredOutcome) -> Self {
        Self {
            time: outcome.time,
            event_occurred: outcome.event_occurred,
        }
    }
}

/// Pocock/Finkelstein-Schoenfeld comparator for right-censored adverse time-to-event endpoints.
///
/// Treatment subject `a` wins when control subject `b` experiences an observed event strictly
/// before their common follow-up (`b_event && a_time > b_time`), loses symmetrically when `a`
/// experiences an earlier observed event (`a_event && b_time > a_time`), and yields
/// [`ComparisonResult::Tie`] when censoring renders ordering indeterminate so evaluation
/// passes to lower-priority tiers.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct RightCensoredComparator {
    /// Optional administrative censoring horizon `c > 0` applied before comparison.
    pub horizon: Option<f64>,
}

impl RightCensoredComparator {
    /// Creates an uncapped right-censored comparator using each subject's observed follow-up time.
    #[verified_engine::verified]
    #[must_use]
    pub fn new() -> Self {
        Self { horizon: None }
    }

    /// Creates a right-censored comparator with an administrative follow-up horizon `c > 0`.
    ///
    /// # Errors
    /// Returns [`ClinicalTrialError::InvalidData`] if `horizon` is non-finite or `<= 0.0`.
    #[verified_engine::verified]
    pub fn with_horizon(horizon: f64) -> Result<Self, ClinicalTrialError> {
        if !horizon.is_finite() || horizon <= 0.0 {
            return Err(ClinicalTrialError::InvalidData(format!(
                "Censoring horizon must be positive and finite, got {horizon}"
            )));
        }
        debug_assert!(horizon > 0.0);
        Ok(Self {
            horizon: Some(horizon),
        })
    }

    #[inline]
    pub(crate) fn compare_raw(
        &self,
        a_time: f64,
        a_event: bool,
        b_time: f64,
        b_event: bool,
    ) -> ComparisonResult {
        debug_assert!(a_time.is_finite() && b_time.is_finite());
        debug_assert!(a_time >= 0.0 && b_time >= 0.0);
        let (ya, da) = apply_horizon(a_time, a_event, self.horizon);
        let (yb, db) = apply_horizon(b_time, b_event, self.horizon);

        if db && ya > yb {
            ComparisonResult::Win
        } else if da && yb > ya {
            ComparisonResult::Loss
        } else {
            ComparisonResult::Tie
        }
    }
}

#[inline]
fn apply_horizon(time: f64, event: bool, horizon: Option<f64>) -> (f64, bool) {
    if let Some(c) = horizon
        && time > c
    {
        return (c, false);
    }
    (time, event)
}

impl OutcomeComparator<CensoredOutcome> for RightCensoredComparator {
    #[verified_engine::verified]
    fn compare(&self, a: &CensoredOutcome, b: &CensoredOutcome) -> ComparisonResult {
        self.compare_raw(
            a.time.as_f64(),
            a.event_occurred,
            b.time.as_f64(),
            b.event_occurred,
        )
    }
}

impl OutcomeComparator<Observation> for RightCensoredComparator {
    #[verified_engine::verified]
    fn compare(&self, a: &Observation, b: &Observation) -> ComparisonResult {
        self.compare_raw(
            a.time.as_f64(),
            a.event_occurred,
            b.time.as_f64(),
            b.event_occurred,
        )
    }
}

/// Pairwise win, loss, and remaining-tie counts resolved at a single priority tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TierCount {
    /// Zero-based index of the endpoint tier in the composite hierarchy.
    pub tier_index: usize,
    /// Number of pairs won by the treatment group at this tier.
    pub wins: usize,
    /// Number of pairs lost by the treatment group at this tier.
    pub losses: usize,
    /// Number of pairs still tied after evaluating this tier.
    pub ties_remaining: usize,
}
