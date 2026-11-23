//! Numerical Integration Strategies.
//!
//! Re-exports `#![no_std]` Newton-Cotes (`Trapezoidal`, `Simpson`) and adaptive
//! `GaussKronrod15` (`G7-K15`) quadrature rules from `engine_diffeq` alongside
//! `ClenshawCurtis` quadrature.

use quadrature::clenshaw_curtis;

pub use engine_diffeq::quadrature::{
    GK15_STACK_SUBINTERVALS, GaussKronrod15, IntegrationResult, Integrator, Simpson, Trapezoidal,
    gk15_panel, trapezoidal_nonuniform, trapezoidal_uniform_grid, trapezoidal_uniform_mapped,
};

/// Clenshaw-Curtis Quadrature Integration.
///
/// Uses the `quadrature` crate for smooth integrands when `std` is available.
#[derive(Debug, Clone, Copy, Default)]
pub struct ClenshawCurtis;

impl Integrator for ClenshawCurtis {
    #[verified_engine::verified]
    fn integrate<F>(&self, f: F, min: f64, max: f64, eps: f64) -> IntegrationResult
    where
        F: Fn(f64) -> f64,
    {
        let result = clenshaw_curtis::integrate(f, min, max, eps);
        IntegrationResult {
            value: result.integral,
            error: result.error_estimate,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[verified_engine::verified]
    fn test_analysis_integration_reexports_all_quadrature_rules() {
        let f = |x: f64| x * x;
        let trap_coarse = Trapezoidal::new(10).integrate(f, 0.0, 10.0, 0.0);
        assert!(trap_coarse.error > 0.0);

        let trap_refined = Trapezoidal::new(10).integrate(f, 0.0, 10.0, 1e-4);
        assert!((trap_refined.value - 1000.0 / 3.0).abs() < 1e-4);

        let simp = Simpson::new(20).integrate(f, 0.0, 10.0, 0.0);
        assert!((simp.value - 1000.0 / 3.0).abs() < 1e-12);

        let gk = GaussKronrod15::new().integrate(|x: f64| (5.0 * x).sin(), 0.0, 4.0, 1e-10);
        let exact = (1.0 - 20.0_f64.cos()) / 5.0;
        assert!((gk.value - exact).abs() < 1e-10);

        let cc = ClenshawCurtis.integrate(f, 0.0, 10.0, 1e-10);
        assert!((cc.value - 1000.0 / 3.0).abs() < 1e-9);
    }
}
