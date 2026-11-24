//! Monotonic threshold sweeps for Receiver Operating Characteristic (`ROC`) and
//! Precision-Recall (`PR`) curves and area-under-curve (`AUC`) summaries.

use math_commons::error::{MathError, MathResult};
use math_commons::math_kernel::types::Dimension;
use serde::{Deserialize, Serialize};

/// Vertex along a Receiver Operating Characteristic (`ROC`) curve.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RocPoint {
    /// Decision score threshold $t$ such that samples with $s_i \ge t$ are predicted positive.
    pub threshold: f64,
    /// False positive rate $\text{FP} / N_-$ in `[0.0, 1.0]`.
    pub fpr: f64,
    /// True positive rate (recall / sensitivity) $\text{TP} / N_+$ in `[0.0, 1.0]`.
    pub tpr: f64,
}

/// Vertex along a Precision-Recall (`PR`) curve.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PrPoint {
    /// Decision score threshold $t$ such that samples with $s_i \ge t$ are predicted positive.
    pub threshold: f64,
    /// Recall (sensitivity) $\text{TP} / N_+$ in `[0.0, 1.0]`.
    pub recall: f64,
    /// Precision (positive predictive value) $\text{TP} / (\text{TP} + \text{FP})$ in `[0.0, 1.0]`.
    pub precision: f64,
}

/// Quadrature convention used when integrating Precision-Recall area under the curve (`PR-AUC`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PrIntegration {
    /// Right-endpoint step sum $\sum_k (R_k - R_{k-1}) P_k$ (Average Precision; avoids optimistic linear interpolation).
    Step,
    /// Piecewise-linear trapezoidal quadrature $\sum_k \frac{1}{2}(R_k - R_{k-1})(P_k + P_{k-1})$.
    Trapezoidal,
}

struct SortedScoreSweep {
    pairs: Vec<(f64, bool)>,
    n_pos: usize,
    n_neg: usize,
}

fn prepare_sorted_scores(scores: &[f64], outcomes: &[bool]) -> MathResult<SortedScoreSweep> {
    if scores.len() != outcomes.len() {
        return Err(MathError::DimensionMismatch {
            expected: Dimension(outcomes.len()),
            actual: Dimension(scores.len()),
        });
    }
    if scores.is_empty() {
        return Err(MathError::NumericalError {
            reason: "Threshold ranking inputs must be non-empty".to_string(),
        });
    }
    let mut n_pos = 0_usize;
    let mut pairs = Vec::with_capacity(scores.len());
    for (&s, &y) in scores.iter().zip(outcomes.iter()) {
        if !s.is_finite() {
            return Err(MathError::NumericalError {
                reason: "Ranking scores must be finite numbers".to_string(),
            });
        }
        if y {
            n_pos += 1;
        }
        pairs.push((s, y));
    }
    let n_neg = scores.len() - n_pos;
    if n_pos == 0 || n_neg == 0 {
        return Err(MathError::NumericalError {
            reason: "ROC and PR curves require both positive and negative outcomes".to_string(),
        });
    }
    pairs.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
    Ok(SortedScoreSweep {
        pairs,
        n_pos,
        n_neg,
    })
}

fn sweep_roc_points(pairs: &[(f64, bool)], n_pos: usize, n_neg: usize) -> Vec<RocPoint> {
    let inv_pos = 1.0 / (n_pos as f64);
    let inv_neg = 1.0 / (n_neg as f64);
    let anchor_threshold = pairs[0].0 + 1.0;
    let mut curve = Vec::with_capacity(pairs.len() + 1);
    curve.push(RocPoint {
        threshold: anchor_threshold,
        fpr: 0.0,
        tpr: 0.0,
    });

    let mut tp = 0_usize;
    let mut fp = 0_usize;
    let mut idx = 0_usize;
    while idx < pairs.len() {
        let score = pairs[idx].0;
        while idx < pairs.len() && pairs[idx].0.to_bits() == score.to_bits() {
            if pairs[idx].1 {
                tp += 1;
            } else {
                fp += 1;
            }
            idx += 1;
        }
        curve.push(RocPoint {
            threshold: score,
            fpr: (fp as f64) * inv_neg,
            tpr: (tp as f64) * inv_pos,
        });
    }
    curve
}

/// Sweeps descending score thresholds (grouping tied scores atomically) to build a monotonic `ROC` curve.
#[verified_engine::verified]
pub fn roc_curve(scores: &[f64], outcomes: &[bool]) -> MathResult<Vec<RocPoint>> {
    let sorted = prepare_sorted_scores(scores, outcomes)?;
    Ok(sweep_roc_points(&sorted.pairs, sorted.n_pos, sorted.n_neg))
}

/// Computes exact trapezoidal `ROC-AUC` (equivalent to the Mann-Whitney $U$ statistic with $0.5$ tie weight).
#[verified_engine::verified]
pub fn roc_auc(scores: &[f64], outcomes: &[bool]) -> MathResult<f64> {
    let curve = roc_curve(scores, outcomes)?;
    let mut area = 0.0;
    for pair in curve.windows(2) {
        let dfpr = pair[1].fpr - pair[0].fpr;
        area += 0.5 * dfpr * (pair[1].tpr + pair[0].tpr);
    }
    Ok(area.clamp(0.0, 1.0))
}

fn sweep_pr_points(pairs: &[(f64, bool)], n_pos: usize) -> Vec<PrPoint> {
    let inv_pos = 1.0 / (n_pos as f64);
    let anchor_threshold = pairs[0].0 + 1.0;
    let mut curve = Vec::with_capacity(pairs.len() + 1);
    curve.push(PrPoint {
        threshold: anchor_threshold,
        recall: 0.0,
        precision: 1.0,
    });

    let mut tp = 0_usize;
    let mut fp = 0_usize;
    let mut idx = 0_usize;
    while idx < pairs.len() {
        let score = pairs[idx].0;
        while idx < pairs.len() && pairs[idx].0.to_bits() == score.to_bits() {
            if pairs[idx].1 {
                tp += 1;
            } else {
                fp += 1;
            }
            idx += 1;
        }
        let predicted_pos = (tp + fp) as f64;
        curve.push(PrPoint {
            threshold: score,
            recall: (tp as f64) * inv_pos,
            precision: (tp as f64) / predicted_pos,
        });
    }
    curve
}

/// Sweeps descending score thresholds (grouping tied scores atomically) to build a Precision-Recall curve.
#[verified_engine::verified]
pub fn precision_recall_curve(scores: &[f64], outcomes: &[bool]) -> MathResult<Vec<PrPoint>> {
    let sorted = prepare_sorted_scores(scores, outcomes)?;
    Ok(sweep_pr_points(&sorted.pairs, sorted.n_pos))
}

/// Computes `PR-AUC` using either [`PrIntegration::Step`] (Average Precision) or
/// [`PrIntegration::Trapezoidal`] quadrature.
#[verified_engine::verified]
pub fn pr_auc(scores: &[f64], outcomes: &[bool], method: PrIntegration) -> MathResult<f64> {
    let curve = precision_recall_curve(scores, outcomes)?;
    let mut area = 0.0;
    for pair in curve.windows(2) {
        let d_recall = pair[1].recall - pair[0].recall;
        match method {
            PrIntegration::Step => {
                area += d_recall * pair[1].precision;
            }
            PrIntegration::Trapezoidal => {
                area += 0.5 * d_recall * (pair[1].precision + pair[0].precision);
            }
        }
    }
    Ok(area.clamp(0.0, 1.0))
}
