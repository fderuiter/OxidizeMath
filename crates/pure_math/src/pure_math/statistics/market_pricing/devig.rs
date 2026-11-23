//! Zero-allocation bookmaker vigorish removal (Multiplicative, Additive, Power, and Shin)
//! and Closing Line Value (CLV) pricing kernels.

use crate::error::KellyError;
use crate::pure_math::statistics::kelly::{Odds, UnitInterval};
use serde::{Deserialize, Serialize};

/// Vigorish-removal method applied to bookmaker decimal odds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DevigMethod {
    /// Proportional normalization $p_i = \pi_i / \sum_j \pi_j$.
    Multiplicative,
    /// Equal-margin subtraction $p_i = \pi_i - (\sum_j \pi_j - 1) / n$.
    Additive,
    /// Power/logarithmic method $p_i = \pi_i^k$ solving $\sum_j \pi_j^k = 1$.
    Power,
    /// Shin (1993) insider-trading equilibrium solving $\sum_j p_j(z) = 1$.
    Shin,
}

/// Computes the raw bookmaker overround $S = \sum_{i=1}^n \frac{1}{o_i}$ for an $n$-outcome market ($n \ge 2$).
#[verified_engine::verified]
pub fn book_overround(odds: &[Odds]) -> Result<f64, KellyError> {
    if odds.len() < 2 {
        return Err(KellyError::InvalidOdds {
            value: odds.len() as f64,
        });
    }
    let mut sum = 0.0;
    for o in odds {
        sum += o.implied_probability();
    }
    Ok(sum)
}

/// Computes the bookmaker vigorish (hold percentage) $\nu = \max(0, 1 - 1 / S)$ bounded in [`UnitInterval`].
#[verified_engine::verified]
pub fn book_vigorish(odds: &[Odds]) -> Result<UnitInterval, KellyError> {
    let s = book_overround(odds)?;
    if s <= 1.0 {
        return Ok(UnitInterval::new_clamped(0.0));
    }
    Ok(UnitInterval::new_clamped(1.0 - 1.0 / s))
}

#[verified_engine::verified]
fn validate_devig_buffers(odds: &[Odds], out_probs: &[UnitInterval]) -> Result<f64, KellyError> {
    if odds.len() != out_probs.len() {
        return Err(KellyError::InvalidOdds {
            value: out_probs.len() as f64,
        });
    }
    book_overround(odds)
}

/// Removes bookmaker vigorish proportionally ($p_i = \pi_i / S$) into `out_probs` without heap allocation.
/// Returns the raw overround $S$.
#[verified_engine::verified]
pub fn devig_multiplicative_into(
    odds: &[Odds],
    out_probs: &mut [UnitInterval],
) -> Result<f64, KellyError> {
    let s = validate_devig_buffers(odds, out_probs)?;
    for (dst, o) in out_probs.iter_mut().zip(odds.iter()) {
        *dst = UnitInterval::new_clamped(o.implied_probability() / s);
    }
    Ok(s)
}

/// Removes bookmaker vigorish via equal-margin subtraction ($p_i = \pi_i - (S - 1) / n$) into `out_probs`.
/// Returns an error if longshot bias drives any outcome probability non-positive.
#[verified_engine::verified]
pub fn devig_additive_into(
    odds: &[Odds],
    out_probs: &mut [UnitInterval],
) -> Result<f64, KellyError> {
    let s = validate_devig_buffers(odds, out_probs)?;
    let shift = (s - 1.0) / (odds.len() as f64);
    for (dst, o) in out_probs.iter_mut().zip(odds.iter()) {
        let p = o.implied_probability() - shift;
        if !(0.0..=1.0).contains(&p) || p <= 0.0 {
            return Err(KellyError::InvalidProbability { value: p });
        }
        *dst = UnitInterval::new_clamped(p);
    }
    Ok(s)
}

#[verified_engine::verified]
fn power_sum(odds: &[Odds], exponent: f64) -> f64 {
    let mut total = 0.0;
    for o in odds {
        total += o.implied_probability().powf(exponent);
    }
    total
}

/// Removes vigorish via the power method ($p_i = \pi_i^k$, $\sum_i \pi_i^k = 1$) using zero-allocation
/// monotone bisection. Returns the fitted power exponent $k > 0$.
#[verified_engine::verified]
pub fn devig_power_into(odds: &[Odds], out_probs: &mut [UnitInterval]) -> Result<f64, KellyError> {
    let s = validate_devig_buffers(odds, out_probs)?;
    if (s - 1.0).abs() <= 1e-12 {
        for (dst, o) in out_probs.iter_mut().zip(odds.iter()) {
            *dst = o.implied_probability_unit();
        }
        return Ok(1.0);
    }
    let (mut lo, mut hi) = if s > 1.0 { (1.0, 4.0) } else { (0.01, 1.0) };
    if s > 1.0 {
        while power_sum(odds, hi) > 1.0 && hi < 1024.0 {
            hi *= 2.0;
        }
    } else {
        while power_sum(odds, lo) < 1.0 && lo > 1e-6 {
            lo *= 0.5;
        }
    }
    for _ in 0..64 {
        let mid = 0.5 * (lo + hi);
        if power_sum(odds, mid) > 1.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let k = 0.5 * (lo + hi);
    let norm = power_sum(odds, k);
    for (dst, o) in out_probs.iter_mut().zip(odds.iter()) {
        *dst = UnitInterval::new_clamped(o.implied_probability().powf(k) / norm);
    }
    Ok(k)
}

#[verified_engine::verified]
fn shin_prob_single(pi: f64, overround: f64, z: f64) -> f64 {
    let denom = 2.0 * (1.0 - z);
    if denom <= 1e-15 {
        return (pi * pi) / overround;
    }
    let disc = (z * z + 4.0 * (1.0 - z) * pi * pi / overround).max(0.0);
    (disc.sqrt() - z) / denom
}

#[verified_engine::verified]
fn shin_sum(odds: &[Odds], overround: f64, z: f64) -> f64 {
    let mut total = 0.0;
    for o in odds {
        total += shin_prob_single(o.implied_probability(), overround, z);
    }
    total
}

/// Removes vigorish via Shin's (1993) insider-trading equilibrium:
///
/// $$p_i(z) = \frac{\sqrt{z^2 + 4(1 - z)\pi_i^2 / S} - z}{2(1 - z)}, \qquad \sum_{i=1}^n p_i(z) = 1$$
///
/// Writes fair probabilities into `out_probs` without heap allocation and returns the estimated
/// insider trading proportion $z \in [0, 1)$ as a [`UnitInterval`].
#[verified_engine::verified]
pub fn devig_shin_into(
    odds: &[Odds],
    out_probs: &mut [UnitInterval],
) -> Result<UnitInterval, KellyError> {
    let s = validate_devig_buffers(odds, out_probs)?;
    if s <= 1.0 {
        devig_multiplicative_into(odds, out_probs)?;
        return Ok(UnitInterval::new_clamped(0.0));
    }
    let mut lo = 0.0_f64;
    let mut hi = 1.0 - 1e-12;
    for _ in 0..64 {
        let mid = 0.5 * (lo + hi);
        if shin_sum(odds, s, mid) > 1.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let z = 0.5 * (lo + hi);
    let total = shin_sum(odds, s, z);
    for (dst, o) in out_probs.iter_mut().zip(odds.iter()) {
        let p = shin_prob_single(o.implied_probability(), s, z) / total;
        *dst = UnitInterval::new_clamped(p);
    }
    Ok(UnitInterval::new_clamped(z))
}
