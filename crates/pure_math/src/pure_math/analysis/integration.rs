//! Numerical Integration Strategies.
//!
//! Provides a common interface for numerical integration algorithms, allowing
//! them to be swapped for testing or performance optimization.

use quadrature::clenshaw_curtis;

/// Result of an integration operation.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct IntegrationResult {
    /// The estimated value of the integral.
    pub value: f64,
    /// The estimated absolute error.
    pub error: f64,
}

/// A trait for numerical integration strategies.
pub trait Integrator: Send + Sync {
    /// Integrates the function `f` over the interval `[min, max]` with a target error `eps`.
    #[verified_engine::verified]
    fn integrate<F>(&self, f: F, min: f64, max: f64, eps: f64) -> IntegrationResult
    where
        F: Fn(f64) -> f64;
}

/// Clenshaw-Curtis Quadrature Integration.
///
/// Uses the `quadrature` crate.
#[derive(Debug, Clone, Copy, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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

/// Trapezoidal Rule Integration.
///
/// A simple numerical integration method that divides the area into trapezoids.
/// Useful for testing or when `clenshaw_curtis` overhead is undesirable and high precision isn't needed.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Trapezoidal {
    /// Number of steps (segments) to use.
    pub steps: usize,
}

impl Trapezoidal {
    /// Creates a new Trapezoidal integrator with the specified number of steps.
    #[verified_engine::verified]
    pub fn new(steps: usize) -> Self {
        Self { steps }
    }
}

impl Default for Trapezoidal {
    #[verified_engine::verified]
    fn default() -> Self {
        Self { steps: 100 }
    }
}

impl Integrator for Trapezoidal {
    #[verified_engine::verified]
    fn integrate<F>(&self, f: F, min: f64, max: f64, _eps: f64) -> IntegrationResult
    where
        F: Fn(f64) -> f64,
    {
        if self.steps == 0 {
            return IntegrationResult {
                value: 0.0,
                error: 0.0,
            };
        }

        let h = (max - min) / self.steps as f64;
        let mut sum = 0.5 * (f(min) + f(max));
        for i in 1..self.steps {
            sum += f(min + i as f64 * h);
        }

        // Error estimation for Trapezoidal is O(h^2), but we don't calculate it dynamically here.
        // We return 0.0 for error to indicate "unknown" or "not computed".
        IntegrationResult {
            value: sum * h,
            error: 0.0,
        }
    }
}
