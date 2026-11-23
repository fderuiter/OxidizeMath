//! # Verified Probability Distributions & Parameter Estimators
//!
//! Stack-allocated continuous and discrete probability distributions with exact densities,
//! log-densities, cumulative probabilities, safeguarded Newton-bisection quantiles, analytical
//! moments, deterministic [`OxidizeRng`](oxidize_core::rng::OxidizeRng) sampling, and Maximum
//! Likelihood (MLE) / Method-of-Moments (MoM) estimators.
//!
//! ## Quick Start
//!
//! ```rust
//! # use pure_math::__doctest_facade as math_explorer;
//! use math_explorer::pure_math::statistics::distributions::{
//!     ContinuousDistribution, DiscreteDistribution, GammaDist, NegativeBinomialDist,
//! };
//! use math_explorer::rng::OxidizeRng;
//!
//! # fn main() -> math_explorer::error::Result<()> {
//! let gamma = GammaDist::new(3.0, 2.0)?;
//! assert!((gamma.mean().unwrap_or(0.0) - 1.5).abs() < 1e-12);
//! let q75 = gamma.quantile(0.75)?;
//! assert!((gamma.cdf(q75) - 0.75).abs() < 1e-10);
//!
//! let mut rng = OxidizeRng::from_seed(12345);
//! let nb = NegativeBinomialDist::new(4.0, 0.4)?;
//! let draw = nb.sample(&mut rng);
//! assert!(nb.pmf(draw) > 0.0);
//! # Ok(())
//! # }
//! ```

mod continuous_bounded_tail;
mod continuous_gamma;
mod continuous_location;
mod continuous_positive;
mod continuous_student;
mod discrete_count;
mod discrete_finite;
pub(crate) mod special;
mod traits;

pub use crate::error::DistributionError;
pub use continuous_bounded_tail::{BetaDist, ChiSquaredDist};
pub use continuous_gamma::GammaDist;
pub use continuous_location::{ExponentialDist, NormalDist, UniformDist};
pub use continuous_positive::{LogNormalDist, WeibullDist};
pub use continuous_student::StudentTDist;
pub use discrete_count::{GeometricDist, NegativeBinomialDist, PoissonDist};
pub use discrete_finite::{BernoulliDist, BinomialDist};
pub use traits::{ContinuousDistribution, DiscreteDistribution};

#[cfg(test)]
mod tests;

// [cite:clinical_trials_statistics]
