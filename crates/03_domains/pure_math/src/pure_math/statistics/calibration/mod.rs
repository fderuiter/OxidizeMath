//! # Probabilistic Calibration, Proper Scoring Rules & Post-Hoc Recalibration
//!
//! Verified primitives for evaluating probabilistic forecasts, diagnosing reliability errors,
//! sweeping classification decision thresholds, and fitting post-hoc recalibrators without
//! domain-crate dependency cycles.
// [cite:calibration]
// [cite:clinical_trials_statistics]
//!
//! ## Core Capabilities
//!
//! - **Proper Scoring Rules ([`scoring`])**: [`brier_score`], exact and binned
//!   [`murphy_decomposition`] ($\text{BS} = \text{REL} - \text{RES} + \text{UNC}$), symmetric
//!   [`binary_log_loss`] and [`multiclass_log_loss`], analytical [`gaussian_crps`], and
//!   zero-allocation [`ensemble_crps_sorted`].
//! - **Calibration Diagnostics ([`diagnostics`])**: Uniform and quantile [`reliability_curve`]s,
//!   [`expected_calibration_error`] (`ECE`), and [`maximum_calibration_error`] (`MCE`).
//! - **Threshold Ranking Curves ([`ranking`])**: Tie-aware monotonic sweeps for [`roc_curve`],
//!   [`roc_auc`], [`precision_recall_curve`], and step or trapezoidal [`pr_auc`].
//! - **Post-Hoc Recalibrators ([`recalibration`])**: Damped Newton-Raphson [`PlattScaler`],
//!   globally convex [`TemperatureScaler`], and zero-allocation Pool Adjacent Violators
//!   ([`IsotonicCalibrator`] / [`fit_pava_sorted_into`]) with piecewise-linear interpolation.
//!
//! ## Quick Start
//!
//! ```rust
//! # use pure_math::__doctest_facade as math_explorer;
//! use math_explorer::primitives::UnitInterval;
//! use math_explorer::pure_math::statistics::calibration::{
//!     BinningStrategy, DEFAULT_LOG_LOSS_EPSILON, TemperatureScaler, binary_log_loss,
//!     expected_calibration_error, murphy_decomposition,
//! };
//!
//! fn main() -> math_explorer::error::Result<()> {
//!     let probs = [
//!         UnitInterval::new(0.05)?,
//!         UnitInterval::new(0.15)?,
//!         UnitInterval::new(0.85)?,
//!         UnitInterval::new(0.95)?,
//!     ];
//!     let outcomes = [false, true, false, true];
//!
//!     let murphy = murphy_decomposition(&probs, &outcomes)?;
//!     assert!(
//!         (murphy.brier_score - (murphy.reliability - murphy.resolution + murphy.uncertainty))
//!             .abs()
//!             < math_explorer::registry::TOLERANCE_STANDARD
//!     );
//!
//!     let nll = binary_log_loss(&probs, &outcomes, DEFAULT_LOG_LOSS_EPSILON)?;
//!     let ece = expected_calibration_error(&probs, &outcomes, 2, BinningStrategy::Uniform)?;
//!     assert!(nll > 0.0 && ece >= 0.0);
//!
//!     let logits = [-3.0, -1.8, 1.8, 3.0];
//!     let scaler = TemperatureScaler::fit_binary(&logits, &outcomes)?;
//!     assert!(scaler.temperature.value() > 1.0);
//!     Ok(())
//! }
//! ```

/// Reliability curves, binning strategies, and Expected / Maximum Calibration Error (`ECE`, `MCE`).
pub mod diagnostics;
/// Monotonic Pool Adjacent Violators Algorithm (`PAVA`) isotonic regression and scratch kernels.
pub mod isotonic;
/// Monotonic threshold sweeps for `ROC` and Precision-Recall curves and area-under-curve metrics.
pub mod ranking;
/// Post-hoc probability recalibrators: Platt scaling, temperature scaling, and `PAVA` isotonic regression.
pub mod recalibration;
/// Strictly proper scoring rules: Brier score, Murphy decomposition, log-loss, and `CRPS`.
pub mod scoring;

#[cfg(test)]
mod tests;

pub use diagnostics::{
    BinningStrategy, CalibrationSummary, ReliabilityBin, calibration_summary,
    expected_calibration_error, maximum_calibration_error, reliability_curve,
    reliability_curve_uniform_into,
};
pub use ranking::{
    PrIntegration, PrPoint, RocPoint, pr_auc, precision_recall_curve, roc_auc, roc_curve,
};
pub use recalibration::{
    IsotonicCalibrator, PavaBlock, PlattConfig, PlattScaler, TemperatureScaler,
    fit_pava_sorted_into, predict_pava_blocks,
};
pub use scoring::{
    BrierDecomposition, DEFAULT_LOG_LOSS_EPSILON, binary_log_loss, brier_score,
    dataset_gaussian_crps, ensemble_crps_sorted, gaussian_crps, log_loss, multiclass_log_loss,
    murphy_decomposition, murphy_decomposition_binned,
};

// [cite:clinical_trials_statistics]
