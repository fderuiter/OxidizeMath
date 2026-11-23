//! Deterministic pricing and zero-allocation antithetic Cholesky Monte Carlo simulation for
//! correlated Same Game Parlays (SGPs).

use super::core::{CorrelationMatrix, Probability};
use super::gaussian::{MAX_STACK_LEGS, evaluate_copula_cdf, sgp_joint_probability_with_scratch};
use super::transforms::fast_norm_inv;
use crate::error::CopulaError;
use math_commons::error::MathError;
use math_commons::math_kernel::types::Dimension;
use nalgebra::DMatrix;
use oxidize_core::rng::OxidizeRng;
use rand::Rng;
use rand_distr::StandardNormal;

/// Summary pricing and simulation metrics for a correlated Same Game Parlay (SGP).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParlayPricingReport {
    /// Correlated joint win probability across all legs.
    pub joint_probability: Probability,
    /// Standard error of `joint_probability` (`0.0` for deterministic quadrature).
    pub standard_error: f64,
    /// Naive independent win probability (`prod_i p_i`).
    pub naive_probability: Probability,
    /// SGP correlation multiplier (`joint_probability / naive_probability`).
    pub correlation_multiplier: f64,
    /// Vig-free fair decimal odds (`1.0 / joint_probability`, or `f64::INFINITY` if zero).
    pub fair_decimal_odds: f64,
}

#[verified_engine::verified]
fn build_report(
    marginals: &[Probability],
    joint_raw: f64,
    standard_error: f64,
) -> Result<ParlayPricingReport, CopulaError> {
    let mut naive_raw = 1.0_f64;
    for p in marginals {
        naive_raw *= p.value();
    }
    let naive_clamped = naive_raw.clamp(0.0, 1.0);
    let joint_clamped = joint_raw.clamp(0.0, 1.0);
    let joint_probability = Probability::new(joint_clamped)?;
    let naive_probability = Probability::new(naive_clamped)?;
    let correlation_multiplier = if naive_clamped > 0.0 {
        joint_clamped / naive_clamped
    } else {
        0.0
    };
    let fair_decimal_odds = if joint_clamped > 0.0 {
        1.0 / joint_clamped
    } else {
        f64::INFINITY
    };

    Ok(ParlayPricingReport {
        joint_probability,
        standard_error: standard_error.max(0.0),
        naive_probability,
        correlation_multiplier,
        fair_decimal_odds,
    })
}

/// Returns the scratch buffer length in `f64` elements recommended for
/// [`simulate_correlated_parlay_into`] (`2 * num_legs`).
#[verified_engine::verified]
pub const fn parlay_simulation_scratch_len(num_legs: usize) -> usize {
    2 * num_legs
}

/// Deterministically prices a correlated parlay using exact boundary reduction and Genz
/// quadrature (zero heap allocations up to [`MAX_STACK_LEGS`] legs).
///
/// # Example
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::nalgebra::DMatrix;
/// use math_explorer::pure_math::statistics::copula::{
///     CorrelationMatrix, Probability, price_correlated_parlay,
/// };
///
/// let legs = [
///     Probability::new(0.55).unwrap(),
///     Probability::new(0.50).unwrap(),
///     Probability::new(0.60).unwrap(),
/// ];
/// let corr = CorrelationMatrix::new(DMatrix::from_row_slice(
///     3,
///     3,
///     &[1.0, 0.45, 0.25, 0.45, 1.0, 0.30, 0.25, 0.30, 1.0],
/// ))
/// .unwrap();
/// let report = price_correlated_parlay(&legs, &corr).unwrap();
/// assert!(report.correlation_multiplier > 1.0);
/// assert!(report.fair_decimal_odds > 1.0);
/// ```
#[verified_engine::verified]
pub fn price_correlated_parlay(
    marginals: &[Probability],
    correlation_matrix: &CorrelationMatrix,
) -> Result<ParlayPricingReport, CopulaError> {
    let joint = evaluate_copula_cdf(marginals, correlation_matrix)?;
    build_report(marginals, joint.value(), 0.0)
}

/// Deterministically prices an arbitrary-leg correlated parlay with zero heap allocations
/// using caller-supplied `scratch`.
#[verified_engine::verified]
pub fn price_correlated_parlay_with_scratch(
    marginals: &[Probability],
    correlation_matrix: &CorrelationMatrix,
    scratch: &mut [f64],
) -> Result<ParlayPricingReport, CopulaError> {
    let joint = sgp_joint_probability_with_scratch(marginals, correlation_matrix, scratch)?;
    build_report(marginals, joint.value(), 0.0)
}

fn run_antithetic_cholesky_pairs(
    cholesky: &DMatrix<f64>,
    thresholds: &[f64],
    eps: &mut [f64],
    n_pairs: usize,
    rng: &mut OxidizeRng,
) -> (u64, u64) {
    let d = thresholds.len();
    let mut both_hits = 0_u64;
    let mut single_hits = 0_u64;

    for _ in 0..n_pairs {
        let mut pos_ok = true;
        let mut neg_ok = true;
        for i in 0..d {
            eps[i] = rng.sample(StandardNormal);
            let mut z_i = 0.0;
            for j in 0..=i {
                z_i += cholesky[(i, j)] * eps[j];
            }
            let bound = thresholds[i];
            pos_ok &= z_i <= bound;
            neg_ok &= -z_i <= bound;
            if !pos_ok && !neg_ok {
                break;
            }
        }
        if pos_ok && neg_ok {
            both_hits += 1;
        } else if pos_ok || neg_ok {
            single_hits += 1;
        }
    }
    (both_hits, single_hits)
}

fn summarize_antithetic_counts(both_hits: u64, single_hits: u64, n_pairs: usize) -> (f64, f64) {
    let m = n_pairs as f64;
    let sum_y = (both_hits as f64) + 0.5 * (single_hits as f64);
    let mean = sum_y / m;
    if n_pairs <= 1 {
        return (mean, 0.0);
    }
    let sum_y_sq = (both_hits as f64) + 0.25 * (single_hits as f64);
    let sample_var = ((sum_y_sq - m * mean * mean) / (m - 1.0)).max(0.0);
    let std_err = (sample_var / m).sqrt();
    (mean, std_err)
}

/// Runs zero-allocation antithetic Cholesky Monte Carlo simulation for a correlated parlay
/// into caller-supplied `scratch` (`scratch.len() >= parlay_simulation_scratch_len(d)`, or
/// `>= d` when `d <= 16`).
///
/// # Example
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::statistics::copula::{
///     Correlation, CorrelationMatrix, Probability, simulate_correlated_parlay_into,
/// };
/// use math_explorer::rng::OxidizeRng;
///
/// let legs = [Probability::new(0.55).unwrap(), Probability::new(0.60).unwrap()];
/// let corr = CorrelationMatrix::bivariate(Correlation::new(0.35).unwrap()).unwrap();
/// let mut rng = OxidizeRng::new(42);
/// let mut scratch = [0.0_f64; 4];
/// let report = simulate_correlated_parlay_into(&legs, &corr, 10_000, &mut rng, &mut scratch).unwrap();
/// assert!(report.joint_probability.value() > report.naive_probability.value());
/// ```
#[verified_engine::verified]
pub fn simulate_correlated_parlay_into(
    marginals: &[Probability],
    correlation_matrix: &CorrelationMatrix,
    n_trials: usize,
    rng: &mut OxidizeRng,
    scratch: &mut [f64],
) -> Result<ParlayPricingReport, CopulaError> {
    let d = correlation_matrix.dimension();
    if marginals.len() != d {
        return Err(CopulaError::Math(MathError::DimensionMismatch {
            expected: Dimension(d),
            actual: Dimension(marginals.len()),
        }));
    }
    if n_trials == 0 {
        return Err(CopulaError::Math(MathError::DimensionMismatch {
            expected: Dimension(1),
            actual: Dimension(0),
        }));
    }
    let min_scratch = if d <= MAX_STACK_LEGS { d } else { 2 * d };
    if scratch.len() < min_scratch {
        return Err(CopulaError::Math(MathError::DimensionMismatch {
            expected: Dimension(2 * d),
            actual: Dimension(scratch.len()),
        }));
    }

    if marginals.iter().any(|p| p.value() == 0.0) {
        return build_report(marginals, 0.0, 0.0);
    }
    if marginals.iter().all(|p| p.value() == 1.0) {
        return build_report(marginals, 1.0, 0.0);
    }

    let n_pairs = n_trials.div_ceil(2);
    let (both_hits, single_hits) = if scratch.len() >= 2 * d {
        let (thresholds, rest) = scratch.split_at_mut(d);
        for (slot, p) in thresholds.iter_mut().zip(marginals.iter()) {
            *slot = fast_norm_inv(p.value());
        }
        run_antithetic_cholesky_pairs(
            correlation_matrix.cholesky_factor(),
            thresholds,
            &mut rest[..d],
            n_pairs,
            rng,
        )
    } else {
        let mut stack_thresholds = [0.0_f64; MAX_STACK_LEGS];
        for (slot, p) in stack_thresholds[..d].iter_mut().zip(marginals.iter()) {
            *slot = fast_norm_inv(p.value());
        }
        run_antithetic_cholesky_pairs(
            correlation_matrix.cholesky_factor(),
            &stack_thresholds[..d],
            &mut scratch[..d],
            n_pairs,
            rng,
        )
    };

    let (mean, std_err) = summarize_antithetic_counts(both_hits, single_hits, n_pairs);
    build_report(marginals, mean, std_err)
}

/// Simulates a correlated Same Game Parlay using antithetic Cholesky Monte Carlo and [`OxidizeRng`].
#[verified_engine::verified]
pub fn simulate_correlated_parlay(
    marginals: &[Probability],
    correlation_matrix: &CorrelationMatrix,
    n_trials: usize,
    rng: &mut OxidizeRng,
) -> Result<ParlayPricingReport, CopulaError> {
    let d = correlation_matrix.dimension();
    if d <= MAX_STACK_LEGS {
        let mut stack_scratch = [0.0_f64; 2 * MAX_STACK_LEGS];
        return simulate_correlated_parlay_into(
            marginals,
            correlation_matrix,
            n_trials,
            rng,
            &mut stack_scratch[..2 * d],
        );
    }
    let mut heap_scratch = vec![0.0_f64; parlay_simulation_scratch_len(d)];
    simulate_correlated_parlay_into(
        marginals,
        correlation_matrix,
        n_trials,
        rng,
        &mut heap_scratch,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    #[verified_engine::verified]
    fn test_parlay_simulation_reproducibility_and_convergence() {
        let legs = [
            Probability::new(0.55).unwrap(),
            Probability::new(0.50).unwrap(),
            Probability::new(0.60).unwrap(),
        ];
        let corr = CorrelationMatrix::new(DMatrix::from_row_slice(
            3,
            3,
            &[1.0, 0.50, 0.25, 0.50, 1.0, 0.35, 0.25, 0.35, 1.0],
        ))
        .unwrap();

        let exact = price_correlated_parlay(&legs, &corr).unwrap();
        let mut rng_a = OxidizeRng::new(0xCAFE_1234);
        let mut rng_b = OxidizeRng::new(0xCAFE_1234);
        let sim_a = simulate_correlated_parlay(&legs, &corr, 40_000, &mut rng_a).unwrap();
        let sim_b = simulate_correlated_parlay(&legs, &corr, 40_000, &mut rng_b).unwrap();

        assert_eq!(sim_a, sim_b);
        assert!(sim_a.standard_error > 0.0 && sim_a.standard_error < 0.004);
        assert_relative_eq!(
            sim_a.joint_probability.value(),
            exact.joint_probability.value(),
            epsilon = 4.0 * sim_a.standard_error
        );
        assert!(exact.correlation_multiplier > 1.0);
        assert_relative_eq!(
            exact.fair_decimal_odds,
            1.0 / exact.joint_probability.value(),
            epsilon = 1e-12
        );
    }
}
