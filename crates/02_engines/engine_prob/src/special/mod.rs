//! `#![no_std]` special functions: Lanczos gamma, beta, incomplete gamma/beta, error functions,
//! and machine-precision standard normal CDF / quantile routines.

pub mod erf;
pub mod gamma;
pub mod incomplete;

pub use erf::{
    erf, erf_inv, erfc, erfc_inv, standard_normal_cdf, standard_normal_ln_pdf, standard_normal_pdf,
    standard_normal_quantile, standard_normal_survival,
};
pub use gamma::{beta, gamma, ln_beta, ln_factorial, ln_gamma};
pub use incomplete::{
    regularized_incomplete_beta, regularized_lower_gamma, regularized_upper_gamma,
};
