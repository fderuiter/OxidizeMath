//! Monotonic Pool Adjacent Violators Algorithm (`PAVA`) isotonic regression with
//! piecewise-linear interpolation and zero-allocation scratch-slice kernels.

use math_commons::error::{MathError, MathResult};
use math_commons::math_kernel::types::Dimension;
use math_commons::primitives::UnitInterval;
use serde::{Deserialize, Serialize};

/// Contiguous pooled monotonic block produced by the Pool Adjacent Violators Algorithm (`PAVA`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PavaBlock {
    /// Minimum input score merged into this block.
    pub min_x: f64,
    /// Maximum input score merged into this block.
    pub max_x: f64,
    /// Pooled isotonic probability estimate $\hat{y}_b \in [0.0, 1.0]$.
    pub value: f64,
    /// Total positive sample weight accumulated in this block.
    pub weight: f64,
}

impl Default for PavaBlock {
    fn default() -> Self {
        Self {
            min_x: 0.0,
            max_x: 0.0,
            value: 0.0,
            weight: 0.0,
        }
    }
}

fn validate_pava_inputs(
    sorted_scores: &[f64],
    sorted_targets: &[f64],
    weights: Option<&[f64]>,
    blocks_scratch: &[PavaBlock],
) -> MathResult<()> {
    let n = sorted_scores.len();
    if n != sorted_targets.len() {
        return Err(MathError::DimensionMismatch {
            expected: Dimension(sorted_targets.len()),
            actual: Dimension(n),
        });
    }
    if let Some(w) = weights
        && w.len() != n
    {
        return Err(MathError::DimensionMismatch {
            expected: Dimension(n),
            actual: Dimension(w.len()),
        });
    }
    if n == 0 {
        return Err(MathError::NumericalError {
            reason: "PAVA isotonic regression requires at least one observation".to_string(),
        });
    }
    if blocks_scratch.len() < n {
        return Err(MathError::DimensionMismatch {
            expected: Dimension(n),
            actual: Dimension(blocks_scratch.len()),
        });
    }
    Ok(())
}

fn pool_pava_core(
    sorted_scores: &[f64],
    sorted_targets: &[f64],
    weights: Option<&[f64]>,
    blocks: &mut [PavaBlock],
) -> MathResult<usize> {
    let mut top = 0_usize;
    for i in 0..sorted_scores.len() {
        let x = sorted_scores[i];
        let y = sorted_targets[i];
        let w = weights.map_or(1.0, |ws| ws[i]);
        if !x.is_finite()
            || !y.is_finite()
            || !(0.0..=1.0).contains(&y)
            || !w.is_finite()
            || w <= 0.0
        {
            return Err(MathError::NumericalError {
                reason: "PAVA requires finite scores, targets in [0, 1], and positive weights"
                    .to_string(),
            });
        }
        if i > 0 && x < sorted_scores[i - 1] {
            return Err(MathError::NumericalError {
                reason: "fit_pava_sorted_into requires non-decreasing sorted_scores".to_string(),
            });
        }
        if top > 0 && blocks[top - 1].max_x.to_bits() == x.to_bits() {
            let prev = &mut blocks[top - 1];
            let total_w = prev.weight + w;
            prev.value = ((prev.weight * prev.value + w * y) / total_w).clamp(0.0, 1.0);
            prev.weight = total_w;
        } else {
            blocks[top] = PavaBlock {
                min_x: x,
                max_x: x,
                value: y.clamp(0.0, 1.0),
                weight: w,
            };
            top += 1;
        }
        while top >= 2 && blocks[top - 2].value >= blocks[top - 1].value {
            let right = blocks[top - 1];
            let left = &mut blocks[top - 2];
            let total_w = left.weight + right.weight;
            left.value =
                ((left.weight * left.value + right.weight * right.value) / total_w).clamp(0.0, 1.0);
            left.weight = total_w;
            left.max_x = right.max_x;
            top -= 1;
        }
    }
    Ok(top)
}

/// Fits monotonic `PAVA` isotonic regression in $O(N)$ time into caller-allocated `blocks_scratch`
/// with zero heap allocations, returning the active block count `M <= sorted_scores.len()`.
#[verified_engine::verified]
pub fn fit_pava_sorted_into(
    sorted_scores: &[f64],
    sorted_targets: &[f64],
    weights: Option<&[f64]>,
    blocks_scratch: &mut [PavaBlock],
) -> MathResult<usize> {
    validate_pava_inputs(sorted_scores, sorted_targets, weights, blocks_scratch)?;
    pool_pava_core(sorted_scores, sorted_targets, weights, blocks_scratch)
}

/// Evaluates a fitted `PAVA` block slice at `score` using $O(\log M)$ binary search, constant
/// plateau values on pooled spans, piecewise-linear interpolation between blocks, and clamped tails.
#[verified_engine::verified]
pub fn predict_pava_blocks(blocks: &[PavaBlock], score: f64) -> MathResult<UnitInterval> {
    if blocks.is_empty() {
        return Err(MathError::NumericalError {
            reason: "Cannot predict with an empty PAVA block slice".to_string(),
        });
    }
    if !score.is_finite() {
        return Err(MathError::NumericalError {
            reason: "Prediction score must be finite".to_string(),
        });
    }
    let prob = if score <= blocks[0].min_x {
        blocks[0].value
    } else if score >= blocks[blocks.len() - 1].max_x {
        blocks[blocks.len() - 1].value
    } else {
        let pos = blocks.partition_point(|b| b.min_x <= score);
        let idx = pos - 1;
        let cur = blocks[idx];
        if score <= cur.max_x {
            cur.value
        } else {
            let next = blocks[idx + 1];
            let span = next.min_x - cur.max_x;
            if span <= f64::EPSILON {
                0.5 * (cur.value + next.value)
            } else {
                let t = ((score - cur.max_x) / span).clamp(0.0, 1.0);
                cur.value + t * (next.value - cur.value)
            }
        }
    };
    UnitInterval::new(prob.clamp(0.0, 1.0)).map_err(MathError::from)
}

/// Non-parametric post-hoc isotonic probability calibrator backed by `PAVA` and piecewise-linear interpolation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IsotonicCalibrator {
    blocks: Vec<PavaBlock>,
}

impl IsotonicCalibrator {
    /// Fits an [`IsotonicCalibrator`] on arbitrary unsorted `(scores, outcomes)` pairs.
    #[verified_engine::verified]
    pub fn fit(scores: &[f64], outcomes: &[bool]) -> MathResult<Self> {
        if scores.len() != outcomes.len() {
            return Err(MathError::DimensionMismatch {
                expected: Dimension(outcomes.len()),
                actual: Dimension(scores.len()),
            });
        }
        if scores.is_empty() {
            return Err(MathError::NumericalError {
                reason: "IsotonicCalibrator::fit requires non-empty inputs".to_string(),
            });
        }
        let mut pairs: Vec<(f64, f64)> = scores
            .iter()
            .zip(outcomes.iter())
            .map(|(&s, &y)| (s, if y { 1.0 } else { 0.0 }))
            .collect();
        if pairs.iter().any(|(s, _)| !s.is_finite()) {
            return Err(MathError::NumericalError {
                reason: "IsotonicCalibrator::fit requires finite scores".to_string(),
            });
        }
        pairs.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));

        let sorted_scores: Vec<f64> = pairs.iter().map(|(s, _)| *s).collect();
        let sorted_targets: Vec<f64> = pairs.iter().map(|(_, y)| *y).collect();
        let mut blocks = vec![PavaBlock::default(); pairs.len()];
        let count = fit_pava_sorted_into(&sorted_scores, &sorted_targets, None, &mut blocks)?;
        blocks.truncate(count);
        let tail_eps = (0.5 / ((pairs.len() + 1) as f64)).min(0.25);
        for block in &mut blocks {
            block.value = block.value.clamp(tail_eps, 1.0 - tail_eps);
        }
        Ok(Self { blocks })
    }

    /// Returns the fitted monotonic `PAVA` blocks.
    #[verified_engine::verified]
    pub fn blocks(&self) -> &[PavaBlock] {
        &self.blocks
    }

    /// Predicts a calibrated [`UnitInterval`] probability for a single scalar `score`.
    #[verified_engine::verified]
    pub fn predict(&self, score: f64) -> MathResult<UnitInterval> {
        predict_pava_blocks(&self.blocks, score)
    }

    /// Evaluates calibrated probabilities for `scores` into `out` with zero heap allocations.
    #[verified_engine::verified]
    pub fn predict_into(&self, scores: &[f64], out: &mut [UnitInterval]) -> MathResult<()> {
        if scores.len() != out.len() {
            return Err(MathError::DimensionMismatch {
                expected: Dimension(scores.len()),
                actual: Dimension(out.len()),
            });
        }
        for (s, dst) in scores.iter().zip(out.iter_mut()) {
            *dst = predict_pava_blocks(&self.blocks, *s)?;
        }
        Ok(())
    }
}
