//! Supervised probabilistic calibration, proper scoring rules, and post-hoc recalibrators for AI models.
//!
//! Unlike [`crate::ai::self_calibration`]—which implements label-free LLM answer-clustering
//! self-consistency (`self_calibration_paper.tex`)—this module evaluates and recalibrates
//! probabilistic classifiers against ground-truth labels using verified primitives from
//! [`pure_math::pure_math::statistics::calibration`].
// [cite:supervised_calibration]
//!
//! ## Example
//!
//! ```rust
//! # use domain_ai::__doctest_facade as math_explorer;
//! use math_explorer::ai::calibration::{BinningStrategy, evaluate_binary_classifier};
//! use math_explorer::nalgebra::DVector;
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let probs = DVector::from_vec(vec![0.1, 0.25, 0.75, 0.9]);
//!     let outcomes = [false, false, true, true];
//!     let report = evaluate_binary_classifier(&probs, &outcomes, 2, BinningStrategy::Uniform)?;
//!     assert!(report.roc_auc > 0.99);
//!     assert!(report.brier.brier_score < 0.05);
//!     Ok(())
//! }
//! ```

use math_commons::primitives::UnitInterval;
use nalgebra::{DMatrix, DVector};

use crate::error::AIError;

pub use pure_math::pure_math::statistics::calibration::{
    BinningStrategy, BrierDecomposition, CalibrationSummary, DEFAULT_LOG_LOSS_EPSILON,
    IsotonicCalibrator, PavaBlock, PlattConfig, PlattScaler, PrIntegration, PrPoint,
    ReliabilityBin, RocPoint, TemperatureScaler, binary_log_loss, brier_score, calibration_summary,
    dataset_gaussian_crps, ensemble_crps_sorted, expected_calibration_error, fit_pava_sorted_into,
    gaussian_crps, log_loss, maximum_calibration_error, multiclass_log_loss, murphy_decomposition,
    murphy_decomposition_binned, pr_auc, precision_recall_curve, predict_pava_blocks,
    reliability_curve, reliability_curve_uniform_into, roc_auc, roc_curve,
};

/// Comprehensive supervised calibration and ranking report for a binary classifier.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BinaryClassifierCalibration {
    /// Exact Murphy decomposition (`brier_score`, `reliability`, `resolution`, `uncertainty`).
    pub brier: BrierDecomposition,
    /// Symmetrically clipped dataset binary log-loss.
    pub log_loss: f64,
    /// Sample-weighted Expected Calibration Error (`ECE`).
    pub ece: f64,
    /// Worst-bin Maximum Calibration Error (`MCE`).
    pub mce: f64,
    /// Receiver Operating Characteristic area under the curve (`ROC-AUC`).
    pub roc_auc: f64,
    /// Average-precision (`Step`) Precision-Recall area under the curve (`PR-AUC`).
    pub pr_auc: f64,
}

fn dvector_to_unit_intervals(probs: &DVector<f64>) -> Result<Vec<UnitInterval>, AIError> {
    let mut out = Vec::with_capacity(probs.len());
    for &p in probs.iter() {
        out.push(UnitInterval::new(p)?);
    }
    Ok(out)
}

/// Evaluates proper scoring rules, calibration errors, and threshold ranking AUCs for a
/// `DVector<f64>` of predicted positive-class probabilities.
#[verified_engine::verified]
pub fn evaluate_binary_classifier(
    predicted_probs: &DVector<f64>,
    outcomes: &[bool],
    num_bins: usize,
    strategy: BinningStrategy,
) -> Result<BinaryClassifierCalibration, AIError> {
    let probs = dvector_to_unit_intervals(predicted_probs)?;
    let brier = murphy_decomposition(&probs, outcomes)?;
    let log_loss = binary_log_loss(&probs, outcomes, DEFAULT_LOG_LOSS_EPSILON)?;
    let summary = calibration_summary(&probs, outcomes, num_bins, strategy)?;
    let raw_scores = predicted_probs.as_slice();
    let roc = roc_auc(raw_scores, outcomes)?;
    let pr = pr_auc(raw_scores, outcomes, PrIntegration::Step)?;
    Ok(BinaryClassifierCalibration {
        brier,
        log_loss,
        ece: summary.ece,
        mce: summary.mce,
        roc_auc: roc,
        pr_auc: pr,
    })
}

/// Fits a [`PlattScaler`] on a `DVector<f64>` of held-out classifier logits and returns calibrated probabilities.
#[verified_engine::verified]
pub fn fit_and_apply_platt(
    cal_logits: &DVector<f64>,
    cal_outcomes: &[bool],
    eval_logits: &DVector<f64>,
) -> Result<(PlattScaler, DVector<f64>), AIError> {
    let scaler = PlattScaler::fit(cal_logits.as_slice(), cal_outcomes)?;
    let mut calibrated = Vec::with_capacity(eval_logits.len());
    for &z in eval_logits.iter() {
        calibrated.push(scaler.predict_proba(z)?.value());
    }
    Ok((scaler, DVector::from_vec(calibrated)))
}

/// Fits a multiclass [`TemperatureScaler`] on a `(n_samples, n_classes)` logit matrix.
#[verified_engine::verified]
pub fn fit_temperature_matrix(
    logits: &DMatrix<f64>,
    targets: &[usize],
) -> Result<TemperatureScaler, AIError> {
    let mut row_major = Vec::with_capacity(logits.nrows() * logits.ncols());
    for r in 0..logits.nrows() {
        for c in 0..logits.ncols() {
            row_major.push(logits[(r, c)]);
        }
    }
    Ok(TemperatureScaler::fit_multiclass(
        &row_major,
        logits.ncols(),
        targets,
    )?)
}

// [cite:supervised_calibration]

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[verified_engine::verified]
    fn test_ai_calibration_adapters_and_temperature_matrix() {
        let cal_logits = DVector::from_vec(vec![-3.0, -1.5, 1.5, 3.0]);
        let outcomes = [false, true, false, true];
        let (platt, calibrated) =
            fit_and_apply_platt(&cal_logits, &outcomes, &cal_logits).expect("platt adapter");
        assert!(platt.scale.is_finite());
        let report =
            evaluate_binary_classifier(&calibrated, &outcomes, 2, BinningStrategy::Uniform)
                .expect("binary report");
        assert!(report.brier.identity_residual() < math_commons::registry::TOLERANCE_STANDARD);

        let mc_logits =
            DMatrix::from_row_slice(3, 3, &[4.0, 0.5, -1.0, 0.2, 3.5, 0.1, -0.5, 0.4, 3.8]);
        let targets = [0_usize, 1, 2];
        let temp = fit_temperature_matrix(&mc_logits, &targets).expect("multiclass temp");
        assert!(temp.temperature.value() > 0.0);
    }
}
