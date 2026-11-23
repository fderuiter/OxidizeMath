#![no_std]
#![doc = include_str!("../README.md")]

#[cfg(feature = "alloc")]
extern crate alloc;
#[cfg(feature = "std")]
extern crate std;

pub mod bootstrap;
pub mod cubature;
pub mod error;
pub mod math;
pub mod variance_reduction;
pub mod workspace;

pub use bootstrap::{
    bca_bootstrap, bca_bootstrap_mean, bca_bootstrap_median, bootstrap_indices,
    percentile_bootstrap, percentile_bootstrap_mean, percentile_bootstrap_median, BootstrapConfig,
    BootstrapInterval, BootstrapMethod,
};
pub use cubature::{
    gaussian_copula_richtmyer_cdf, genz_semidefinite_richtmyer_qmc, integrate_hyperrect,
    integrate_unit_cube, mvn_genz_richtmyer_cdf, CubatureConfig, CubatureEstimate, CubatureRule,
};
pub use error::ProbError;
pub use variance_reduction::{
    control_variate_from_slices, estimate_antithetic_normal, estimate_antithetic_uniform,
    estimate_control_variate, estimate_self_normalized_is, self_normalized_is_from_slices,
    simulate_correlated_bernoulli_antithetic, simulate_orthant_antithetic_col_major,
    ControlVariateEstimate, CorrelatedBernoulliSummary, ImportanceSamplingEstimate,
    MonteCarloEstimate,
};
pub use workspace::SolverWorkspace;

#[cfg(feature = "alloc")]
pub use workspace::OwnedSolverWorkspace;

#[cfg(test)]
mod tests;
