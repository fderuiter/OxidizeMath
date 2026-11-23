//! Numerical Integration Strategies.
//!
//! Provides a common interface for numerical integration algorithms, allowing
//! them to be swapped for testing or performance optimization.

use quadrature::clenshaw_curtis;

/// Result of an integration operation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IntegrationResult {
    /// The estimated value of the integral.
    pub value: f64,
    /// The estimated absolute error.
    pub error: f64,
}

/// Numerical quadrature interface approximating definite integrals $I(f) = \int_a^b f(x)\,dx \approx \sum_{i=1}^n w_i f(x_i)$.
///
/// Reference: Davis & Rabinowitz (1984), *Methods of Numerical Integration*, DOI: 10.1016/C2013-0-10566-1; [`docs/math_specs/pure_math.md`](../../../../../docs/math_specs/pure_math.md).
///
/// # Stability
/// Positive quadrature weights $w_i > 0$ summing to $b - a$ guarantee bounded rounding amplification $\sum_i |w_i| = |b - a|$.
///
/// # Domain of Validity
/// Valid for Riemann-integrable functions $f$ on finite intervals $(a, b)$ without interior non-integrable singularities.
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
