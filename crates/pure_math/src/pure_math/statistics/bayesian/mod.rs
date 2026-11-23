//! # Conjugate Bayesian Inference & Zero-Allocation MCMC
//!
//! Exact conjugate Bayesian models ([`BetaBinomial`], [`GammaPoisson`], [`NormalNormal`],
//! [`NormalInverseGamma`], [`DirichletMultinomial`]), equal-tailed and Highest Posterior Density
//! (HPD) [`CredibleInterval`] solvers, analytical [`BayesFactor`] model selection on Jeffreys'
//! scale, and pre-allocated zero-allocation Metropolis-Hastings samplers ([`MetropolisHastings1D`],
//! [`MetropolisHastingsND`]).
//!
//! ## Quick Start
//!
//! ```rust
//! # use pure_math::__doctest_facade as math_explorer;
//! use math_explorer::pure_math::statistics::bayesian::{
//!     BetaBinomial, MetropolisHastings1D,
//! };
//! use math_explorer::rng::OxidizeRng;
//!
//! # fn main() -> math_explorer::error::Result<()> {
//! let prior = BetaBinomial::uniform();
//! let posterior = prior.update(14, 20)?;
//! let ci95 = posterior.credible_interval(0.95)?;
//! assert!(ci95.contains(posterior.posterior_mean()));
//!
//! let mut rng = OxidizeRng::from_seed(42);
//! let mut sampler = MetropolisHastings1D::new(0.5, 0.12)?;
//! let mut chain = [0.0_f64; 512];
//! let summary = sampler.sample_into(
//!     |theta| {
//!         if theta <= 0.0 || theta >= 1.0 {
//!             f64::NEG_INFINITY
//!         } else {
//!             14.0 * theta.ln() + 6.0 * (1.0 - theta).ln()
//!         }
//!     },
//!     200,
//!     &mut chain,
//!     &mut rng,
//! )?;
//! assert!((summary.mean - posterior.posterior_mean()).abs() < 0.06);
//! # Ok(())
//! # }
//! ```

mod conjugate_continuous;
mod conjugate_discrete;
mod dirichlet;
mod intervals_and_bf;
mod mcmc;

pub use crate::error::BayesianError;
pub use conjugate_continuous::{NormalInverseGamma, NormalNormal};
pub use conjugate_discrete::{BetaBinomial, GammaPoisson};
pub use dirichlet::DirichletMultinomial;
pub use intervals_and_bf::{
    BayesFactor, CredibleInterval, JeffreysEvidence, equal_tailed_credible_interval,
    hpd_credible_interval,
};
pub use mcmc::{
    McmcSummary, MetropolisHastings1D, MetropolisHastingsND, empirical_credible_interval,
};

#[cfg(test)]
mod tests;

// [cite:clinical_trials_statistics]
