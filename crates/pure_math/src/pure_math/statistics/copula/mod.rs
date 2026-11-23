//! # Gaussian Copula & Same Game Parlay Engine
//!
//! This module implements deterministic Gaussian copula pricing and zero-allocation Cholesky
//! Monte Carlo simulation for correlated events in sports analytics, particularly multi-leg
//! Same Game Parlays (SGPs).
//!
//! ## The Problem: Correlated Events
//!
//! Under naive independence, leg win probabilities multiply:
//! ```text
//! P(A ∩ B ∩ C) = P(A) × P(B) × P(C)
//! ```
//!
//! In real games, parlay legs share latent game-state factors:
//! - Quarterback passing touchdowns and wide receiver receiving yards exhibit positive correlation
//! - High-usage "hero ball" scoring bursts (`1%` tail probability) can negatively correlate with team wins
//!
//! ## Architecture
//!
//! 1. **Validated Correlation Structure ([`CorrelationMatrix`])**: Enforces symmetry, unit
//!    diagonal, finite bounds, and positive-semidefiniteness via eigenvalue verification while
//!    caching the lower-triangular Cholesky factor for reuse across pricing and simulation calls.
//! 2. **Exact Boundary Reduction**: Short-circuits `0.0` marginals to `0.0`, reduces `1.0`
//!    marginals to lower-dimensional sub-copulas, and returns `C(u_1) = u_1` for single-leg inputs.
//! 3. **20-Point Genz Bivariate Quadrature ([`bivariate`])**: Splits Plackett angular integration
//!    and Genz asymptotic tail integration at `|rho| = 0.925` while enforcing Fréchet-Hoeffding bounds.
//! 4. **Multi-Leg Richtmyer QMC ([`GaussianCopula`])**: Evaluates 3+ leg copulas via Genz
//!    separation-of-variables with greedy variable prioritization over an antithetic Baker-tent
//!    Richtmyer quasi-Monte Carlo sequence (`<= 16` legs stack-buffered, arbitrary legs via scratch).
//! 5. **Zero-Allocation Antithetic Parlay Simulator ([`parlay`])**: Estimates multi-leg win rates,
//!    standard errors, SGP correlation multipliers, and fair decimal odds using [`OxidizeRng`](oxidize_core::rng::OxidizeRng).
//!
//! ## Example: Hero Ball & Multi-Leg SGP Pricing
//!
//! ```rust
//! # use pure_math::__doctest_facade as math_explorer;
//! use math_explorer::nalgebra::DMatrix;
//! use math_explorer::pure_math::statistics::copula::{
//!     Correlation, CorrelationMatrix, Probability, price_correlated_parlay,
//!     sgp_joint_probability,
//! };
//!
//! // Event A: Star scores 50+ points (99th-percentile performance => 1% tail probability)
//! let p_star_50 = Probability::new(0.01).unwrap();
//! // Event B: Team wins (60% base probability)
//! let p_team_win = Probability::new(0.60).unwrap();
//!
//! // Negative correlation (-0.15): hero-ball usage slightly hurts team efficiency
//! let corr_2 = CorrelationMatrix::bivariate(Correlation::new(-0.15).unwrap()).unwrap();
//! let joint_2 = sgp_joint_probability(&[p_star_50, p_team_win], &corr_2).unwrap();
//! assert!(joint_2.value() < 0.01 * 0.60);
//!
//! // Three-leg positively correlated SGP priced into fair decimal odds
//! let legs_3 = [
//!     Probability::new(0.55).unwrap(),
//!     Probability::new(0.50).unwrap(),
//!     Probability::new(0.60).unwrap(),
//! ];
//! let corr_3 = CorrelationMatrix::new(DMatrix::from_row_slice(
//!     3,
//!     3,
//!     &[1.0, 0.50, 0.25, 0.50, 1.0, 0.35, 0.25, 0.35, 1.0],
//! ))
//! .unwrap();
//! let report = price_correlated_parlay(&legs_3, &corr_3).unwrap();
//! assert!(report.correlation_multiplier > 1.0);
//! assert!(report.fair_decimal_odds < 1.0 / report.naive_probability.value());
//! ```
//!
//! ## References
//!
//! - Genz, A. (1992). *Numerical Computation of Multivariate Normal Probabilities*. JCGS, 1(2), 141–149.
//! - Genz, A. (2004). *Numerical Computation of Rectangular Bivariate and Trivariate Normal and t Probabilities*. Statistics and Computing, 14, 251–260.
//! - Joe, H. (2014). *Dependence Modeling with Copulas*. Chapman and Hall/CRC.
//! - Nelsen, R. B. (2006). *An Introduction to Copulas* (2nd ed.). Springer.

pub mod bivariate;
pub mod core;
pub mod gaussian;
mod multivariate;
pub mod parlay;
pub mod transforms;

pub use bivariate::{RHO_SPLIT, bivariate_copula_cdf, bivariate_normal_cdf};
pub use core::{Correlation, CorrelationMatrix, Probability};
pub use gaussian::{
    GaussianCopula, MAX_STACK_LEGS, sgp_joint_probability, sgp_joint_probability_with_scratch,
};
pub use parlay::{
    ParlayPricingReport, parlay_simulation_scratch_len, price_correlated_parlay,
    price_correlated_parlay_with_scratch, simulate_correlated_parlay,
    simulate_correlated_parlay_into,
};
pub use transforms::{
    NormalTransform, ProbabilityTransform, inverse_standard_normal, standard_normal_cdf,
};

// [cite:clinical_trials_statistics]
