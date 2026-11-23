//! Zero-allocation `#![no_std]` numerical quadrature traits, Newton-Cotes rules, and Gauss-Kronrod 15.
// [cite:engine_diffeq]

/// Adaptive 7-point Gauss / 15-point Kronrod (`G7-K15`) quadrature.
pub mod gauss_kronrod;
/// Composite `Trapezoidal`, `Simpson`, and discrete sample trapezoidal kernels.
pub mod newton_cotes;

use serde::{Deserialize, Serialize};

/// Value and estimated absolute error bound produced by a numerical quadrature rule.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct IntegrationResult {
    /// Approximated definite integral $\int_a^b f(x)\,dx$.
    pub value: f64,
    /// Estimated upper bound on $|\hat{I} - I_{\text{true}}|$.
    pub error: f64,
}

/// Strategy trait for definite 1D numerical quadrature over $[a, b]$.
pub trait Integrator: Send + Sync {
    /// Integrates `f(x)` from `a` to `b` targeting absolute error tolerance `eps`.
    #[verified_engine::verified]
    fn integrate<F>(&self, f: F, a: f64, b: f64, eps: f64) -> IntegrationResult
    where
        F: Fn(f64) -> f64;
}

pub use gauss_kronrod::{GK15_STACK_SUBINTERVALS, GaussKronrod15, gk15_panel};
pub use newton_cotes::{
    Simpson, Trapezoidal, trapezoidal_nonuniform, trapezoidal_uniform_grid,
    trapezoidal_uniform_mapped,
};
