//! Preallocated Monte Carlo estimation and variance-reduction workspace.
//!
//! Provides calibrated [`MonteCarloEstimate`] summaries and a preallocated
//! [`MonteCarloWorkspace`] supporting:
//! - Standard IID sampling
//! - Antithetic variates
//! - Optimal-$\hat{\beta}^*$ control variates
//! - Self-normalized importance sampling with Kong's exact Effective Sample Size (`ESS`)
//! - Stratified Latin Hypercube sampling (LHS)
//! - Cranley-Patterson randomized Halton and Joe-Kuo Sobol quasi-Monte Carlo
//!
//! ## Example
//!
//! ```rust
//! # use pure_math::__doctest_facade as math_explorer;
//! use math_explorer::prelude::OxidizeRng;
//! use math_explorer::pure_math::statistics::monte_carlo::MonteCarloWorkspace;
//!
//! fn main() -> math_explorer::error::Result<()> {
//!     let mut ws = MonteCarloWorkspace::new(512, 1)?;
//!     let mut rng = OxidizeRng::new(7);
//!     let est = ws.estimate_antithetic(512, &mut rng, |u| u[0].exp())?;
//!     let exact = std::f64::consts::E - 1.0;
//!     assert!((est.mean - exact).abs() < 0.02);
//!     assert!(est.standard_error > 0.0);
//!     Ok(())
//! }
//! ```

mod error;
mod qmc_estimators;
mod stats;
pub(crate) mod workspace;

pub use error::MonteCarloError;
pub use stats::{
    MonteCarloEstimate, effective_sample_size, normal_cdf_fast, normal_inv_cdf_fast,
    two_sided_normal_z,
};
pub use workspace::{MonteCarloWorkspace, SamplingMethod};

#[cfg(test)]
mod tests;

// [cite:clinical_trials_statistics]
