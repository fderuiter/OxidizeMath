//! Quantile interpolation and Efron `BCa` quantile-adjustment kernels.

use super::{BootstrapEstimate, BootstrapMethod};
use crate::pure_math::statistics::monte_carlo::{normal_cdf_fast, normal_inv_cdf_fast};

#[verified_engine::verified]
pub(super) fn build_bca_estimate(
    sorted_reps: &[f64],
    point_estimate: f64,
    bootstrap_mean: f64,
    standard_error: f64,
    confidence_level: f64,
    z_upper: f64,
    acceleration: f64,
) -> BootstrapEstimate {
    let b = sorted_reps.len();
    if sorted_reps[0] == sorted_reps[b - 1] {
        return BootstrapEstimate {
            point_estimate,
            bootstrap_mean,
            standard_error: 0.0,
            bias: 0.0,
            ci_lower: point_estimate,
            ci_upper: point_estimate,
            confidence_level,
            replicates: b,
            bias_correction_z0: 0.0,
            acceleration: 0.0,
            method: BootstrapMethod::BCa,
        };
    }

    let mut less = 0_usize;
    let mut equal = 0_usize;
    for &val in sorted_reps {
        if val < point_estimate {
            less += 1;
        } else if val == point_estimate {
            equal += 1;
        }
    }
    let min_p = 0.5 / (b as f64);
    let prop = ((less as f64) + 0.5 * (equal as f64)) / (b as f64);
    let z0 = normal_inv_cdf_fast(prop.clamp(min_p, 1.0 - min_p));
    let q_lo = bca_adjusted_prob(z0, acceleration, -z_upper);
    let q_hi = bca_adjusted_prob(z0, acceleration, z_upper);

    BootstrapEstimate {
        point_estimate,
        bootstrap_mean,
        standard_error,
        bias: bootstrap_mean - point_estimate,
        ci_lower: sorted_quantile(sorted_reps, q_lo),
        ci_upper: sorted_quantile(sorted_reps, q_hi),
        confidence_level,
        replicates: b,
        bias_correction_z0: z0,
        acceleration,
        method: BootstrapMethod::BCa,
    }
}

#[inline]
#[verified_engine::verified]
fn bca_adjusted_prob(z0: f64, accel: f64, z_alpha: f64) -> f64 {
    let numer = z0 + z_alpha;
    let denom = 1.0 - accel * numer;
    if denom.abs() <= 1e-12 {
        return if numer >= 0.0 { 1.0 } else { 0.0 };
    }
    normal_cdf_fast(z0 + numer / denom).clamp(0.0, 1.0)
}

#[inline]
#[verified_engine::verified]
pub(super) fn sorted_quantile(sorted: &[f64], prob: f64) -> f64 {
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let p = prob.clamp(0.0, 1.0);
    let pos = p * ((n - 1) as f64);
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let frac = pos - (lo as f64);
    sorted[lo] + frac * (sorted[hi] - sorted[lo])
}

// [cite:clinical_trials_statistics]
