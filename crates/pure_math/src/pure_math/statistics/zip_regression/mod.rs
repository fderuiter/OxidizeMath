//! # Zero-Inflated Poisson (ZIP) Regression
//!
//! A sophisticated statistical framework for modeling count data with excess zeros,
//! commonly used in sports analytics, epidemiology, and other fields where
//! structural zeros are present.
//!
//! ## Overview
//!
//! The Zero-Inflated Poisson (ZIP) model addresses a fundamental problem in count
//! modeling: **overdispersion** caused by excess zeros. Unlike standard Poisson
//! distributions where mean = variance, ZIP models account for situations where
//! zeros occur more frequently than a Poisson model would predict.
//!
//! ## Mathematical Framework
//!
//! The ZIP distribution is a **mixture model** combining two processes:
//!
//! ### Process A: Structural Zeros
//! A Bernoulli trial determines if the subject is in an "Always Zero" state:
//! - Probability ρ (rho): Structural zero (e.g., player didn't play)
//! - Probability (1-ρ): Active state (can generate counts)
//!
//! ### Process B: Poisson Counts
//! If active, counts follow a Poisson distribution with rate λ (lambda):
//! - Can still produce zeros (chance zeros)
//! - Can produce positive counts
//!
//! ### Probability Mass Function
//!
//! ```text
//! P(Y = 0) = ρ + (1-ρ)e^(-λ)           (structural OR chance zero)
//! P(Y = k) = (1-ρ)(λ^k e^(-λ))/k!    (for k > 0)
//! ```
//!
//! ### Statistical Properties
//!
//! - **Mean**: E\[Y\] = (1-ρ)λ
//! - **Variance**: Var\[Y\] = (1-ρ)λ(1 + ρλ)
//! - **Overdispersion**: Var\[Y\] > E\[Y\] when ρ > 0
//!
//! ## ZIP Regression
//!
//! In regression settings, both ρ and λ are modeled as functions of covariates:
//!
//! ### Count Model (Poisson Rate)
//! ```text
//! log(λᵢ) = β₀ + β₁x₁ᵢ + ... + βₚxₚᵢ
//! ```
//! Uses a **log link** to ensure λ > 0
//!
//! ### Zero-Inflation Model
//! ```text
//! logit(ρᵢ) = α₀ + α₁z₁ᵢ + ... + αᵧzᵧᵢ
//! ```
//! Uses a **logit link** to ensure ρ ∈ [0, 1]
//!
//! ## Sports Analytics Application
//!
//! ### The "Median Attack" Strategy
//!
//! In sports betting, books often price lines based on the **mean** of a distribution.
//! However, ZIP distributions are heavily **right-skewed**:
//!
//! - If ρ is high (high structural zero probability), the **median** may be 0
//! - But the **mean** could be 0.8 or 0.9 due to the long tail
//! - Book sets line at 0.5 based on mean
//! - True median is 0 → "Under" is statistically advantageous
//!
//! This creates a betting edge by exploiting the mean-median discrepancy.
//!
//! ## Example Usage
//!
//! ```rust
//! # use pure_math::__doctest_facade as math_explorer;
//! use math_explorer::pure_math::statistics::zip_regression::{
//!     ZipDistribution, ZipParams, ZeroInflation, PoissonRate, Count
//! };
//!
//! // Create a ZIP distribution
//! let params = ZipParams::from_values(0.3, 2.0).unwrap();
//! let dist = ZipDistribution::new(params);
//!
//! // Compute probabilities
//! let prob_zero = dist.pmf(Count::new(0));
//! let prob_one = dist.pmf(Count::new(1));
//!
//! // Get statistical properties
//! let mean = dist.mean();          // (1-0.3)*2.0 = 1.4
//! let variance = dist.variance();  // Shows overdispersion
//!
//! println!("Mean: {}, Variance: {}", mean, variance);
//! assert!(variance > mean);  // Overdispersion
//! ```
//!
//! ## Simple Regression Example
//!
//! ```rust
//! # use pure_math::__doctest_facade as math_explorer;
//! use math_explorer::pure_math::statistics::zip_regression::{simple_zip_fit, Count};
//!
//! // Observed player block counts over 10 games
//! let blocks = vec![
//!     Count::new(0), Count::new(0), Count::new(1),
//!     Count::new(0), Count::new(2), Count::new(0),
//!     Count::new(1), Count::new(0), Count::new(0),
//!     Count::new(3)
//! ];
//!
//! // Fit simple ZIP model using method of moments
//! let params = simple_zip_fit(&blocks).unwrap();
//!
//! println!("Estimated ρ (zero-inflation): {}", params.rho.value());
//! println!("Estimated λ (Poisson rate): {}", params.lambda.value());
//! ```
//!
//! ## References
//!
//! - Lambert, D. (1992). "Zero-Inflated Poisson Regression, with an Application to
//!   Defects in Manufacturing." *Technometrics*, 34(1), 1-14.
//! - Ridout, M., Hinde, J., & Demétrio, C. G. (2001). "A Score Test for Testing a
//!   Zero-Inflated Poisson Regression Model Against Zero-Inflated Negative Binomial
//!   Alternatives." *Biometrics*, 57(1), 219-223.

pub mod core;
pub mod distribution;
pub mod link_functions;
pub mod regression;

pub use core::{Count, PoissonRate, ZeroInflation, ZipParams};
pub use distribution::ZipDistribution;
pub use link_functions::{LogLink, LogitLink};
pub use regression::{ZipRegression, simple_zip_fit};

// [cite:clinical_trials_statistics]
