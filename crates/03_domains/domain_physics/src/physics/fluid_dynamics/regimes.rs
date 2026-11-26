//! Flow regime classification strategies.
//!
//! This module implements the **Strategy Pattern** for classifying fluid flow
//! as Laminar, Transitional, or Turbulent based on the Reynolds number ($Re$).
//! Different geometries (pipes, flat plates, airfoils) have different critical
//! Reynolds numbers, making a single hardcoded function insufficient.

/// Classification of flow regimes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlowRegime {
    /// Ordered flow with parallel streamlines.
    Laminar,
    /// Unstable flow mixing laminar and turbulent characteristics.
    Transitional,
    /// Chaotic flow with strong mixing and eddies.
    Turbulent,
}

/// A strategy for classifying flow regimes.
pub trait FlowClassifier {
    /// Determines the flow regime for a given Reynolds number.
    #[verified_engine::verified]
    fn classify(&self, re: f64) -> FlowRegime;
}

/// Standard classifier for internal pipe flow.
///
/// * $Re < 2000$: Laminar
/// * $2000 \le Re \le 4000$: Transitional
/// * $Re > 4000$: Turbulent
#[derive(Debug, Clone, Copy, Default)]
pub struct PipeFlowClassifier;

impl FlowClassifier for PipeFlowClassifier {
    #[verified_engine::verified]
    fn classify(&self, re: f64) -> FlowRegime {
        if re < 2000.0 {
            FlowRegime::Laminar
        } else if re <= 4000.0 {
            FlowRegime::Transitional
        } else {
            FlowRegime::Turbulent
        }
    }
}

/// Classifier for external flow over a flat plate.
///
/// * $Re < 5 \times 10^5$: Laminar
/// * $Re \ge 5 \times 10^5$: Turbulent
///
/// Note: This assumes a smooth plate and zero pressure gradient.
#[derive(Debug, Clone, Copy, Default)]
pub struct FlatPlateClassifier;

impl FlowClassifier for FlatPlateClassifier {
    #[verified_engine::verified]
    fn classify(&self, re: f64) -> FlowRegime {
        if re < 500_000.0 {
            FlowRegime::Laminar
        } else {
            FlowRegime::Turbulent
        }
    }
}
