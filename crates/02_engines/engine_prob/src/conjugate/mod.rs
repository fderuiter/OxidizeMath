//! Zero-allocation closed-form conjugate Bayesian families with exact credible intervals,
//! posterior predictive distributions, log marginal likelihoods, and Bayes factors.

pub mod discrete_families;
pub mod normal_families;

pub use discrete_families::{
    BetaBinomial, DirichletMultinomial, GammaPoisson, dirichlet_bayes_factor,
    dirichlet_log_marginal_likelihood, dirichlet_update_into,
};
pub use normal_families::{NormalInverseGamma, NormalNormal};
