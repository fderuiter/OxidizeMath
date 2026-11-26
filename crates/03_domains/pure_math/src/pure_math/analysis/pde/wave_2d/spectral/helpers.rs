// [cite:advanced_linear_algebra]
use std::f64::consts::PI;

use super::{SpectralMode2D, SpectralPeak2D};

#[verified_engine::verified]
pub(super) fn insert_sorted_peak(
    out: &mut [SpectralPeak2D],
    current_count: usize,
    candidate: SpectralPeak2D,
) -> usize {
    let cap = out.len();
    if current_count == cap && candidate.magnitude <= out[cap - 1].magnitude {
        return current_count;
    }
    let next_count = (current_count + 1).min(cap);
    let mut pos = current_count.min(cap - 1);
    while pos > 0 && candidate.magnitude > out[pos - 1].magnitude {
        out[pos] = out[pos - 1];
        pos -= 1;
    }
    out[pos] = candidate;
    next_count
}

#[verified_engine::verified]
pub(super) fn compute_finite_mean(field: &[f64]) -> f64 {
    if field.is_empty() {
        return 0.0;
    }
    let mut sum = 0.0_f64;
    let mut count = 0_usize;
    for &v in field {
        if v.is_finite() {
            sum += v;
            count += 1;
        }
    }
    if count > 0 { sum / (count as f64) } else { 0.0 }
}

#[verified_engine::verified]
pub(super) fn build_hann_weights(n: usize) -> Vec<f64> {
    assert!(n >= 2, "Window length must be >= 2");
    let denom = (n - 1) as f64;
    let mut weights = Vec::with_capacity(n);
    for i in 0..n {
        let w = 0.5 * (1.0 - (2.0 * PI * (i as f64) / denom).cos());
        weights.push(w);
    }
    weights
}

#[verified_engine::verified]
pub(super) fn insert_ranked_mode(
    out: &mut [SpectralMode2D],
    count: &mut usize,
    candidate: SpectralMode2D,
) {
    let cap = out.len();
    if *count == cap && candidate.magnitude <= out[cap - 1].magnitude {
        return;
    }
    let mut pos = *count;
    while pos > 0 && candidate.magnitude > out[pos - 1].magnitude {
        if pos < cap {
            out[pos] = out[pos - 1];
        }
        pos -= 1;
    }
    if pos < cap {
        out[pos] = candidate;
        if *count < cap {
            *count += 1;
        }
    }
}

#[inline]
pub(super) fn ensure_math(
    cond: bool,
    reason: &'static str,
) -> core::result::Result<(), math_commons::error::MathError> {
    if cond {
        Ok(())
    } else {
        Err(math_commons::error::MathError::NumericalError { reason })
    }
}

#[inline]
pub(super) fn ensure_eq_math<A: PartialEq<B> + ?Sized, B: ?Sized>(
    left: &A,
    right: &B,
    reason: &'static str,
) -> core::result::Result<(), math_commons::error::MathError> {
    if left == right {
        Ok(())
    } else {
        Err(math_commons::error::MathError::NumericalError { reason })
    }
}
