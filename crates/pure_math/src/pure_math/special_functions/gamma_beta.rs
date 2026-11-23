pub use engine_prob::special::{
    erf, erf_inv, erfc, erfc_inv, ln_beta, ln_gamma, regularized_incomplete_beta,
    regularized_lower_gamma, regularized_upper_gamma,
};

/// Gamma Function $\Gamma(x) = \int_0^\infty t^{x-1} e^{-t} dt$.
///
/// Evaluates the `#![no_std]` Lanczos $g = 7, N = 9$ approximation from [`engine_prob::special`].
#[verified_engine::verified]
pub fn gamma_function(x: f64) -> f64 {
    engine_prob::gamma(x)
}

/// Beta Function $B(m, n) = \int_0^1 t^{m-1} (1-t)^{n-1} dt$.
///
/// Evaluates $\exp(\ln \Gamma(m) + \ln \Gamma(n) - \ln \Gamma(m + n))$ via [`engine_prob::special`].
#[verified_engine::verified]
pub fn beta_function(m: f64, n: f64) -> f64 {
    engine_prob::beta(m, n)
}

/// Computes factorial using Gamma function.
/// $n! = \Gamma(n+1)$
#[verified_engine::verified]
pub fn factorial(n: u64) -> f64 {
    gamma_function(n as f64 + 1.0)
}
