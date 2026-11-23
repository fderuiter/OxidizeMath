#![no_std]
#![doc = include_str!("../README.md")]

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(test)]
extern crate std;

/// Hidden doctest shim mapping `math_explorer::engine_prob` to this crate.
#[doc(hidden)]
pub mod __doctest_facade {
    pub use crate as engine_prob;
}

pub mod calibration;
pub mod conjugate;
pub mod distributions;
pub mod error;
pub mod special;
pub mod transport;

#[cfg(feature = "alloc")]
pub use calibration::conformal_quantile;
pub use calibration::{
    EValue, conformal_p_value, conformal_prediction_mask_into, conformal_quantile_into,
    conformal_quantile_sorted,
};

pub use conjugate::{
    BetaBinomial, DirichletMultinomial, GammaPoisson, NormalInverseGamma, NormalNormal,
    dirichlet_bayes_factor, dirichlet_log_marginal_likelihood, dirichlet_update_into,
};
pub use distributions::{
    Bernoulli, Beta, Binomial, Categorical, ChiSquared, ContinuousDist, DiscreteDist, Exponential,
    Gamma, Normal, Poisson, StudentsT, Uniform, discrete_posterior_into, next_f64_open01,
    next_f64_unit, normalize_log_weights_into,
};
pub use error::ProbError;
pub use special::{
    beta, erf, erf_inv, erfc, erfc_inv, gamma, ln_beta, ln_factorial, ln_gamma,
    regularized_incomplete_beta, regularized_lower_gamma, regularized_upper_gamma,
    standard_normal_cdf, standard_normal_ln_pdf, standard_normal_pdf, standard_normal_quantile,
    standard_normal_survival,
};
pub use transport::{
    SinkhornConfig, SinkhornReport, sinkhorn_log_domain_into, sinkhorn_scratch_len,
    wasserstein_1d_into, wasserstein_1d_sorted, wasserstein_1d_weights,
};
#[cfg(feature = "alloc")]
pub use transport::{sinkhorn_log_domain, wasserstein_1d};

// [cite:clinical_trials_statistics]
// [cite:self_calibration_paper]
