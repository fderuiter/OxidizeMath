//! Reliability curve binning and calibration error summaries (`ECE` and `MCE`).

use math_commons::error::{MathError, MathResult};
use math_commons::primitives::UnitInterval;
use serde::{Deserialize, Serialize};

use super::scoring::{brier_score, validate_binary_inputs};

/// Partitioning rule used when grouping predicted probabilities into reliability bins.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BinningStrategy {
    /// Equal-width intervals $[b / B, (b + 1) / B]$ across $[0, 1]$.
    Uniform,
    /// Equal-mass empirical quantile slices over sorted predicted probabilities.
    Quantile,
}

/// Summary statistics for a single reliability diagram bin.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ReliabilityBin {
    /// Zero-based index of the bin in `[0, num_bins)`.
    pub bin_index: usize,
    /// Inclusive lower boundary of the bin in `[0.0, 1.0]`.
    pub lower: f64,
    /// Upper boundary of the bin in `[0.0, 1.0]`.
    pub upper: f64,
    /// Number of forecast-outcome pairs assigned to this bin.
    pub count: usize,
    /// Average predicted probability $\bar{p}_b$ within the bin (`0.0` when `count == 0`).
    pub mean_confidence: f64,
    /// Empirical positive event rate $\bar{y}_b$ within the bin (`0.0` when `count == 0`).
    pub empirical_accuracy: f64,
    /// Absolute calibration discrepancy $|\bar{p}_b - \bar{y}_b|$ (`0.0` when `count == 0`).
    pub calibration_gap: f64,
}

impl Default for ReliabilityBin {
    fn default() -> Self {
        Self {
            bin_index: 0,
            lower: 0.0,
            upper: 1.0,
            count: 0,
            mean_confidence: 0.0,
            empirical_accuracy: 0.0,
            calibration_gap: 0.0,
        }
    }
}

/// Aggregate calibration error metrics across a reliability partition.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CalibrationSummary {
    /// Sample-weighted Expected Calibration Error $\sum_b \frac{n_b}{N} |\bar{p}_b - \bar{y}_b|$.
    pub ece: f64,
    /// Worst-case non-empty bin Maximum Calibration Error $\max_{b: n_b > 0} |\bar{p}_b - \bar{y}_b|$.
    pub mce: f64,
    /// Exact dataset Brier score $\frac{1}{N}\sum_{i=1}^N (p_i - y_i)^2$.
    pub brier_score: f64,
}

fn validate_num_bins(num_bins: usize) -> MathResult<()> {
    if num_bins == 0 {
        return Err(MathError::NumericalError {
            reason: "Number of calibration bins must be at least 1",
        });
    }
    Ok(())
}

fn finalize_uniform_bins(bins: &mut [ReliabilityBin]) {
    for bin in bins.iter_mut() {
        if bin.count > 0 {
            let inv = 1.0 / (bin.count as f64);
            bin.mean_confidence *= inv;
            bin.empirical_accuracy *= inv;
            bin.calibration_gap = (bin.mean_confidence - bin.empirical_accuracy).abs();
        }
    }
}

/// Populates equal-width reliability bins into caller-provided `bins_out` with zero heap allocation.
///
/// Returns the number of initialized bins (`bins_out.len()`).
#[verified_engine::verified]
pub fn reliability_curve_uniform_into(
    probs: &[UnitInterval],
    outcomes: &[bool],
    bins_out: &mut [ReliabilityBin],
) -> MathResult<usize> {
    validate_binary_inputs(probs, outcomes)?;
    let num_bins = bins_out.len();
    validate_num_bins(num_bins)?;

    let inv_bins = 1.0 / (num_bins as f64);
    for (b, bin) in bins_out.iter_mut().enumerate() {
        *bin = ReliabilityBin {
            bin_index: b,
            lower: (b as f64) * inv_bins,
            upper: if b + 1 == num_bins {
                1.0
            } else {
                ((b + 1) as f64) * inv_bins
            },
            count: 0,
            mean_confidence: 0.0,
            empirical_accuracy: 0.0,
            calibration_gap: 0.0,
        };
    }

    let b_f64 = num_bins as f64;
    for (p, &y) in probs.iter().zip(outcomes.iter()) {
        let raw_idx = (p.value() * b_f64).floor() as usize;
        let idx = raw_idx.min(num_bins - 1);
        bins_out[idx].count += 1;
        bins_out[idx].mean_confidence += p.value();
        if y {
            bins_out[idx].empirical_accuracy += 1.0;
        }
    }

    finalize_uniform_bins(bins_out);
    Ok(num_bins)
}

fn build_quantile_bins(
    probs: &[UnitInterval],
    outcomes: &[bool],
    num_bins: usize,
) -> Vec<ReliabilityBin> {
    let mut pairs: Vec<(f64, bool)> = probs
        .iter()
        .zip(outcomes.iter())
        .map(|(p, &y)| (p.value(), y))
        .collect();
    pairs.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));

    let n = pairs.len();
    let mut bins = Vec::with_capacity(num_bins);
    for b in 0..num_bins {
        let start = (b * n) / num_bins;
        let end = ((b + 1) * n) / num_bins;
        if start >= end {
            bins.push(ReliabilityBin {
                bin_index: b,
                lower: 0.0,
                upper: 0.0,
                ..ReliabilityBin::default()
            });
            continue;
        }
        let slice = &pairs[start..end];
        let count = slice.len();
        let sum_p: f64 = slice.iter().map(|(p, _)| *p).sum();
        let sum_y = slice.iter().filter(|(_, y)| *y).count() as f64;
        let mean_conf = sum_p / (count as f64);
        let emp_acc = sum_y / (count as f64);
        bins.push(ReliabilityBin {
            bin_index: b,
            lower: slice[0].0,
            upper: slice[count - 1].0,
            count,
            mean_confidence: mean_conf,
            empirical_accuracy: emp_acc,
            calibration_gap: (mean_conf - emp_acc).abs(),
        });
    }
    bins
}

/// Computes reliability curve bins under either [`BinningStrategy::Uniform`] or
/// [`BinningStrategy::Quantile`].
#[verified_engine::verified]
pub fn reliability_curve(
    probs: &[UnitInterval],
    outcomes: &[bool],
    num_bins: usize,
    strategy: BinningStrategy,
) -> MathResult<Vec<ReliabilityBin>> {
    validate_binary_inputs(probs, outcomes)?;
    validate_num_bins(num_bins)?;
    match strategy {
        BinningStrategy::Uniform => {
            let mut bins = vec![ReliabilityBin::default(); num_bins];
            reliability_curve_uniform_into(probs, outcomes, &mut bins)?;
            Ok(bins)
        }
        BinningStrategy::Quantile => Ok(build_quantile_bins(probs, outcomes, num_bins)),
    }
}

fn ece_mce_from_bins(bins: &[ReliabilityBin], total_samples: usize) -> (f64, f64) {
    let n = total_samples as f64;
    let mut ece = 0.0;
    let mut mce = 0.0_f64;
    for bin in bins {
        if bin.count > 0 {
            ece += ((bin.count as f64) / n) * bin.calibration_gap;
            mce = mce.max(bin.calibration_gap);
        }
    }
    (ece, mce)
}

/// Computes Expected Calibration Error (`ECE`) $\sum_{b: n_b > 0} \frac{n_b}{N} |\bar{p}_b - \bar{y}_b|$.
#[verified_engine::verified]
pub fn expected_calibration_error(
    probs: &[UnitInterval],
    outcomes: &[bool],
    num_bins: usize,
    strategy: BinningStrategy,
) -> MathResult<f64> {
    let bins = reliability_curve(probs, outcomes, num_bins, strategy)?;
    Ok(ece_mce_from_bins(&bins, probs.len()).0)
}

/// Computes Maximum Calibration Error (`MCE`) $\max_{b: n_b > 0} |\bar{p}_b - \bar{y}_b|$.
#[verified_engine::verified]
pub fn maximum_calibration_error(
    probs: &[UnitInterval],
    outcomes: &[bool],
    num_bins: usize,
    strategy: BinningStrategy,
) -> MathResult<f64> {
    let bins = reliability_curve(probs, outcomes, num_bins, strategy)?;
    Ok(ece_mce_from_bins(&bins, probs.len()).1)
}

/// Computes `ECE`, `MCE`, and exact Brier score in a single summary struct.
#[verified_engine::verified]
pub fn calibration_summary(
    probs: &[UnitInterval],
    outcomes: &[bool],
    num_bins: usize,
    strategy: BinningStrategy,
) -> MathResult<CalibrationSummary> {
    let bins = reliability_curve(probs, outcomes, num_bins, strategy)?;
    let (ece, mce) = ece_mce_from_bins(&bins, probs.len());
    let bs = brier_score(probs, outcomes)?;
    Ok(CalibrationSummary {
        ece,
        mce,
        brier_score: bs,
    })
}
