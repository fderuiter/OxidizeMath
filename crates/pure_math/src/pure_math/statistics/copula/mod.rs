//! # Gaussian Copula for Same Game Parlays
//!
//! This module implements copula-based methods for modeling correlated events
//! in sports betting, particularly for pricing Same Game Parlays (SGPs).
//!
//! ## The Problem: Correlated Events
//!
//! In standard probability, if events are independent:
//! ```text
//! P(A ∩ B) = P(A) × P(B)
//! ```
//!
//! However, in sports, many events are correlated:
//! - If a QB throws 4 TDs, the WR likely has high yardage
//! - If a player scores 50 points ("hero ball"), the team may be less likely to win
//!
//! Simply multiplying probabilities leads to massive pricing errors.
//!
//! ## The Solution: Copulas
//!
//! Copulas are functions that "glue" separate marginal distributions together
//! while preserving their correlation structure.
//!
//! ### Step 1: Probability Integral Transform
//!
//! Transform each variable to a uniform \[0,1\] distribution:
//! ```text
//! U = F(X)
//! ```
//! where F is the CDF of X.
//!
//! ### Step 2: Gaussian Copula
//!
//! Map the uniform variables through a multivariate normal distribution:
//! ```text
//! C(u₁, u₂; ρ) = Φ_ρ(Φ⁻¹(u₁), Φ⁻¹(u₂))
//! ```
//!
//! where:
//! - Φ⁻¹ is the inverse standard normal CDF
//! - Φ_ρ is the bivariate normal CDF with correlation ρ
//!
//! ### Step 3: Joint Probability
//!
//! The copula gives us the joint cumulative probability, adjusted for correlation:
//! - **ρ > 0** (positive correlation): Increases joint probability
//! - **ρ < 0** (negative correlation): Decreases joint probability
//!
//! ## Example: "Hero Ball" Analysis
//!
//! ```rust
//! # use pure_math::__doctest_facade as math_explorer;
//! use math_explorer::pure_math::statistics::copula::{
//!     sgp_joint_probability, Probability, CorrelationMatrix, Correlation
//! };
//!
//! // Event A: Luka scores 50+ points (99th percentile)
//! let p_luka_50 = Probability::new(0.99).unwrap();
//!
//! // Event B: Mavs win (60% base probability)
//! let p_mavs_win = Probability::new(0.60).unwrap();
//!
//! // Historical data shows negative correlation: -0.15
//! // (High individual usage slightly hurts team efficiency)
//! let rho = Correlation::new(-0.15).unwrap();
//! let corr_matrix = CorrelationMatrix::bivariate(rho).unwrap();
//!
//! // Calculate true joint probability with copula
//! let joint_prob = sgp_joint_probability(
//!     &[p_luka_50, p_mavs_win],
//!     &corr_matrix
//! ).unwrap();
//!
//! // Compare to naive independence assumption
//! let naive_prob = 0.99 * 0.60;  // = 0.594 (59.4%)
//! println!("Naive probability: {:.4}", naive_prob);
//! println!("Copula-adjusted probability: {:.4}", joint_prob.value());
//!
//! // The copula adjusts for negative correlation, reducing the true probability
//! ```
//!
//! ## Correlation Interpretation
//!
//! - **Positive Correlation (ρ > 0)**:
//!   - Example: QB yards + WR yards
//!   - Copula increases joint probability
//!   - Sportsbook should offer lower odds (smaller payout)
//!
//! - **Negative Correlation (ρ < 0)**:
//!   - Example: RB1 rushing yards + RB2 rushing yards (cannibalization)
//!   - Copula decreases joint probability
//!   - Sportsbook should offer higher odds (larger payout)
//!
//! ## References
//!
//! - Joe, H. (2014). *Dependence Modeling with Copulas*. Chapman and Hall/CRC.
//! - Nelsen, R. B. (2006). *An Introduction to Copulas* (2nd ed.). Springer.

pub mod core;
pub mod gaussian;
mod qmc;
pub mod transforms;

pub use core::{Correlation, CorrelationMatrix, Probability};
pub use gaussian::{GaussianCopula, sgp_joint_probability};
pub use transforms::{
    NormalTransform, ProbabilityTransform, inverse_standard_normal, standard_normal_cdf,
};

// [cite:clinical_trials_statistics]
