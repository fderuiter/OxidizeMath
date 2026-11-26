//! Multi-outcome bookmaker vig removal (`Multiplicative`, `Additive`, `Power`, and `Shin`).

use super::MAX_MARKET_OUTCOMES;
use super::overround::{MarketOverround, implied_probabilities_into};
use crate::error::MarketPricingError;
use crate::pure_math::analysis::roots::{Brent, RootFinder};
use crate::pure_math::statistics::kelly::Odds;
use math_commons::registry::TOLERANCE_HIGH;

/// Bookmaker vig-removal model used to recover true fair probabilities from quoted odds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum DevigMethod {
    /// Proportional scaling (`p_i = pi_i / sum(pi)`), assuming constant margin across all odds.
    Multiplicative,
    /// Equal-margin subtraction (`p_i = pi_i - (sum(pi) - 1) / n`), with simplex clamping.
    Additive,
    /// Constant-exponent calibration solving `sum(pi_i^k) = 1` via `Brent` root-finding.
    Power,
    /// Shin (1991, 1993) insider-trading equilibrium calibration solving `sum(p_i(z)) = 1`.
    Shin,
}

/// Diagnostic emitted when raw `Additive` devigging produces non-positive probabilities on
/// longshots, requiring clamping and renormalization onto the probability simplex.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SimplexViolationDiagnostic {
    /// Count of outcomes whose raw additive probability was `<= 0.0`.
    pub violated_count: usize,
    /// Index of the outcome with the minimum (most negative) raw additive probability.
    pub worst_outcome_index: usize,
    /// Minimum raw additive probability before simplex clamping (e.g. `-0.014328`).
    pub min_raw_probability: f64,
    /// Uniform margin subtracted per outcome (`(booksum - 1.0) / outcome_count`).
    pub margin_per_outcome: f64,
}

/// Calibration parameter, solver telemetry, and simplex diagnostic produced by a devigging run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DevigMetadata {
    /// Devigging model applied.
    pub method: DevigMethod,
    /// Raw market booksum, overround, and turnover margin.
    pub market: MarketOverround,
    /// Model scalar (`1 / booksum` for Multiplicative, shift for Additive, `k` for Power, `z` for Shin).
    pub parameter: f64,
    /// Number of bracketed solver evaluations (`0` for closed-form solutions).
    pub solver_iterations: usize,
    /// Present when raw `Additive` probabilities fell `<= 0.0` on one or more longshot outcomes.
    pub simplex_violation: Option<SimplexViolationDiagnostic>,
}

/// Removes vig proportionally (`p_i = pi_i / booksum`) into `probs_out`.
#[verified_engine::verified]
pub fn devig_multiplicative_into(
    odds: &[Odds],
    probs_out: &mut [f64],
) -> Result<DevigMetadata, MarketPricingError> {
    let market = implied_probabilities_into(odds, probs_out)?;
    let inv_sum = 1.0 / market.booksum;
    let mut i = 0;
    while i < probs_out.len() {
        probs_out[i] *= inv_sum;
        i += 1;
    }
    Ok(DevigMetadata {
        method: DevigMethod::Multiplicative,
        market,
        parameter: inv_sum,
        solver_iterations: 0,
        simplex_violation: None,
    })
}

/// Computes both raw (`pi_i - (S - 1) / n`) and simplex-clamped `Additive` probabilities,
/// surfacing [`SimplexViolationDiagnostic`] whenever a longshot raw probability falls `<= 0.0`.
#[verified_engine::verified]
pub fn devig_additive_detailed_into(
    odds: &[Odds],
    raw_probs_out: &mut [f64],
    clamped_probs_out: &mut [f64],
) -> Result<DevigMetadata, MarketPricingError> {
    if clamped_probs_out.len() != odds.len() {
        return Err(MarketPricingError::SliceLengthMismatch {
            expected: odds.len(),
            actual: clamped_probs_out.len(),
        });
    }
    let market = implied_probabilities_into(odds, raw_probs_out)?;
    let shift = market.overround / (odds.len() as f64);
    let mut violated_count = 0usize;
    let mut worst_idx = 0usize;
    let mut min_raw = f64::INFINITY;
    let mut positive_sum = 0.0;
    let mut i = 0;
    while i < raw_probs_out.len() {
        let raw = raw_probs_out[i] - shift;
        raw_probs_out[i] = raw;
        if raw < min_raw {
            min_raw = raw;
            worst_idx = i;
        }
        if raw <= 0.0 {
            violated_count += 1;
            clamped_probs_out[i] = 0.0;
        } else {
            clamped_probs_out[i] = raw;
            positive_sum += raw;
        }
        i += 1;
    }
    normalize_slice(clamped_probs_out, positive_sum)?;
    let simplex_violation = if violated_count > 0 {
        Some(SimplexViolationDiagnostic {
            violated_count,
            worst_outcome_index: worst_idx,
            min_raw_probability: min_raw,
            margin_per_outcome: shift,
        })
    } else {
        None
    };
    Ok(DevigMetadata {
        method: DevigMethod::Additive,
        market,
        parameter: shift,
        solver_iterations: 0,
        simplex_violation,
    })
}

/// Removes vig via `Additive` equal-margin subtraction, clamping onto the simplex if any
/// longshot goes non-positive while surfacing [`SimplexViolationDiagnostic`] in [`DevigMetadata`].
#[verified_engine::verified]
pub fn devig_additive_into(
    odds: &[Odds],
    probs_out: &mut [f64],
) -> Result<DevigMetadata, MarketPricingError> {
    let mut raw_buf = [0.0_f64; MAX_MARKET_OUTCOMES];
    let n = odds.len();
    if n > MAX_MARKET_OUTCOMES {
        return Err(MarketPricingError::TooManyOutcomes {
            count: n,
            max_supported: MAX_MARKET_OUTCOMES,
        });
    }
    devig_additive_detailed_into(odds, &mut raw_buf[..n], probs_out)
}

#[verified_engine::verified]
fn normalize_slice(probs: &mut [f64], sum: f64) -> Result<(), MarketPricingError> {
    if !sum.is_finite() || sum <= 0.0 {
        return Err(MarketPricingError::NonPositiveOverround { booksum: sum });
    }
    let inv = 1.0 / sum;
    let mut i = 0;
    while i < probs.len() {
        probs[i] *= inv;
        i += 1;
    }
    Ok(())
}

#[verified_engine::verified]
fn power_sum_residual(implied: &[f64], k: f64) -> f64 {
    let mut sum = 0.0;
    let mut i = 0;
    while i < implied.len() {
        sum += implied[i].powf(k);
        i += 1;
    }
    sum - 1.0
}

#[verified_engine::verified]
fn power_upper_bound(implied: &[f64]) -> f64 {
    let mut max_pi = 0.0_f64;
    let mut i = 0;
    while i < implied.len() {
        if implied[i] > max_pi {
            max_pi = implied[i];
        }
        i += 1;
    }
    let neg_ln_max = (-max_pi.ln()).max(1e-12);
    let analytic = 2.0 * (implied.len() as f64).ln() / neg_ln_max;
    analytic.max(2.0)
}

/// Removes vig via the constant-exponent `Power` method (`sum(pi_i^k) = 1`) solved with [`Brent`].
#[verified_engine::verified]
pub fn devig_power_into(
    odds: &[Odds],
    probs_out: &mut [f64],
) -> Result<DevigMetadata, MarketPricingError> {
    let mut implied = [0.0_f64; MAX_MARKET_OUTCOMES];
    let n = odds.len();
    if n > MAX_MARKET_OUTCOMES {
        return Err(MarketPricingError::TooManyOutcomes {
            count: n,
            max_supported: MAX_MARKET_OUTCOMES,
        });
    }
    if probs_out.len() != n {
        return Err(MarketPricingError::SliceLengthMismatch {
            expected: n,
            actual: probs_out.len(),
        });
    }
    let market = implied_probabilities_into(odds, &mut implied[..n])?;
    let slice = &implied[..n];
    let (k, evals) = if market.overround.abs() <= TOLERANCE_HIGH {
        (1.0, 0usize)
    } else {
        let (lo, hi) = if market.overround > 0.0 {
            (1.0, power_upper_bound(slice))
        } else {
            (0.0, 1.0)
        };
        let count = core::cell::Cell::new(0usize);
        let solver = Brent::new(100, 1e-12);
        let root = solver.find_root(
            |exp| {
                count.set(count.get() + 1);
                power_sum_residual(slice, exp)
            },
            lo,
            hi,
        )?;
        (root, count.get())
    };
    let mut sum = 0.0;
    let mut i = 0;
    while i < n {
        let p = slice[i].powf(k);
        probs_out[i] = p;
        sum += p;
        i += 1;
    }
    normalize_slice(probs_out, sum)?;
    Ok(DevigMetadata {
        method: DevigMethod::Power,
        market,
        parameter: k,
        solver_iterations: evals,
        simplex_violation: None,
    })
}

#[verified_engine::verified]
fn shin_prob_rationalized(pi: f64, booksum: f64, z: f64) -> f64 {
    let ratio = (pi * pi) / booksum;
    let radicand = (z * z + 4.0 * (1.0 - z) * ratio).max(0.0);
    (2.0 * ratio) / (radicand.sqrt() + z)
}

#[verified_engine::verified]
fn shin_sum_residual(implied: &[f64], booksum: f64, z: f64) -> f64 {
    let mut sum = 0.0;
    let mut i = 0;
    while i < implied.len() {
        sum += shin_prob_rationalized(implied[i], booksum, z);
        i += 1;
    }
    sum - 1.0
}

/// Removes vig via Shin's (1991, 1993) insider-trading equilibrium model.
///
/// Uses the exact closed-form `z` for 2-way markets and solves the rationalized probability
/// conservation equation on `[0.0, 1.0]` via [`Brent`] for `N >= 3` outcomes.
#[verified_engine::verified]
pub fn devig_shin_into(
    odds: &[Odds],
    probs_out: &mut [f64],
) -> Result<DevigMetadata, MarketPricingError> {
    let mut implied = [0.0_f64; MAX_MARKET_OUTCOMES];
    let n = odds.len();
    if n > MAX_MARKET_OUTCOMES {
        return Err(MarketPricingError::TooManyOutcomes {
            count: n,
            max_supported: MAX_MARKET_OUTCOMES,
        });
    }
    if probs_out.len() != n {
        return Err(MarketPricingError::SliceLengthMismatch {
            expected: n,
            actual: probs_out.len(),
        });
    }
    let market = implied_probabilities_into(odds, &mut implied[..n])?;
    if market.booksum < 1.0 - TOLERANCE_HIGH {
        return Err(MarketPricingError::NonPositiveOverround {
            booksum: market.booksum,
        });
    }
    let slice = &implied[..n];
    let s = market.booksum;
    let (z, evals) = if market.overround.abs() <= TOLERANCE_HIGH {
        (0.0, 0usize)
    } else if n == 2 {
        let diff_sq = (slice[0] - slice[1]) * (slice[0] - slice[1]);
        let z_exact = ((s - 1.0) * (s - diff_sq)) / (s * (1.0 - diff_sq));
        (z_exact.clamp(0.0, 1.0 - 1e-15), 0usize)
    } else {
        let count = core::cell::Cell::new(0usize);
        let solver = Brent::new(100, 1e-12);
        let root = solver.find_root(
            |z_cand| {
                count.set(count.get() + 1);
                shin_sum_residual(slice, s, z_cand)
            },
            0.0,
            1.0,
        )?;
        (root, count.get())
    };
    let mut sum = 0.0;
    let mut i = 0;
    while i < n {
        let p = shin_prob_rationalized(slice[i], s, z);
        probs_out[i] = p;
        sum += p;
        i += 1;
    }
    normalize_slice(probs_out, sum)?;
    Ok(DevigMetadata {
        method: DevigMethod::Shin,
        market,
        parameter: z,
        solver_iterations: evals,
        simplex_violation: None,
    })
}

/// Dispatches vig removal for `method` and writes calibrated fair probabilities into `probs_out`.
#[verified_engine::verified]
pub fn devig_into(
    odds: &[Odds],
    method: DevigMethod,
    probs_out: &mut [f64],
) -> Result<DevigMetadata, MarketPricingError> {
    match method {
        DevigMethod::Multiplicative => devig_multiplicative_into(odds, probs_out),
        DevigMethod::Additive => devig_additive_into(odds, probs_out),
        DevigMethod::Power => devig_power_into(odds, probs_out),
        DevigMethod::Shin => devig_shin_into(odds, probs_out),
    }
}

/// Converts fair probabilities `(0.0, 1.0)` into vig-free [`Odds`] (`1.0 / p_i`).
#[verified_engine::verified]
pub fn fair_odds_from_probabilities_into(
    probs: &[f64],
    fair_odds_out: &mut [Odds],
) -> Result<(), MarketPricingError> {
    if fair_odds_out.len() != probs.len() {
        return Err(MarketPricingError::SliceLengthMismatch {
            expected: probs.len(),
            actual: fair_odds_out.len(),
        });
    }
    let mut i = 0;
    while i < probs.len() {
        let p = probs[i];
        if !p.is_finite() || p <= 0.0 || p >= 1.0 {
            return Err(MarketPricingError::InvalidProbability { value: p });
        }
        let o =
            Odds::new(1.0 / p).map_err(|_| MarketPricingError::InvalidProbability { value: p })?;
        fair_odds_out[i] = o;
        i += 1;
    }
    Ok(())
}

/// Converts fair probabilities `[0.0, 1.0]` into raw decimal odds (`f64::INFINITY` if `p == 0.0`).
#[verified_engine::verified]
pub fn fair_decimal_odds_into(
    probs: &[f64],
    decimal_odds_out: &mut [f64],
) -> Result<(), MarketPricingError> {
    if decimal_odds_out.len() != probs.len() {
        return Err(MarketPricingError::SliceLengthMismatch {
            expected: probs.len(),
            actual: decimal_odds_out.len(),
        });
    }
    let mut i = 0;
    while i < probs.len() {
        let p = probs[i];
        if !p.is_finite() || !(0.0..=1.0).contains(&p) {
            return Err(MarketPricingError::InvalidProbability { value: p });
        }
        decimal_odds_out[i] = if p == 0.0 { f64::INFINITY } else { 1.0 / p };
        i += 1;
    }
    Ok(())
}

// [cite:clinical_trials_statistics]
