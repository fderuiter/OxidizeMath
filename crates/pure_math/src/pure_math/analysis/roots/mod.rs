//! Root finding algorithms for scalar equations and multivariate nonlinear systems.
//!
//! This module implements the **Strategy Pattern** for scalar root finding via [`RootFinder`]
//! ([`Bisection`], [`Brent`], [`NewtonRaphson`], and [`Secant`]), plus [`MultivariateNewton`]
//! for fixed-size stack vectors and pre-allocated dynamic state vectors.

mod bracketing;
mod multivariate;
mod open;

#[cfg(test)]
mod tests;

pub use bracketing::{Bisection, Brent};
pub use multivariate::MultivariateNewton;
pub use open::{NewtonRaphson, Secant};

use thiserror::Error;

/// Copy, zero-allocation errors that can occur during numerical root finding and analysis.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum AnalysisError {
    /// The algorithm failed to converge within the maximum number of iterations.
    /// Contains the best guess (or final iterate / residual summary) so far.
    #[error("Algorithm failed to converge. Best guess: {0}")]
    ConvergenceError(f64),
    /// Endpoints do not bracket a sign change or evaluated to non-finite values.
    #[error("Root not bracketed: f({min})={f_min}, f({max})={f_max}; signs must differ")]
    RootNotBracketed {
        /// Lower bound input.
        min: f64,
        /// Objective value at `min`.
        f_min: f64,
        /// Upper bound input.
        max: f64,
        /// Objective value at `max`.
        f_max: f64,
    },
    /// Derivative or secant slope magnitude fell below the safe division threshold.
    #[error("Derivative too small at x={x}: {derivative}")]
    DerivativeTooSmall {
        /// Iterate where the derivative vanished.
        x: f64,
        /// Evaluated derivative or secant denominator value.
        derivative: f64,
    },
    /// Non-finite interval bounds, initial guesses, or invalid solver tolerance were supplied.
    #[error("Invalid solver bracket [{min}, {max}] or tolerance")]
    InvalidBracket {
        /// Lower bound or first guess input.
        min: f64,
        /// Upper bound or second guess input.
        max: f64,
    },
    /// Invalid solver configuration, non-finite interior evaluation, or singular Jacobian.
    #[error("Invalid parameters: {0}")]
    InvalidParameters(&'static str),
}

/// Strategy for finding scalar roots of a function $f(x) = 0$.
///
/// # Examples
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::analysis::roots::{
///     Bisection, Brent, NewtonRaphson, RootFinder, Secant,
/// };
///
/// fn main() -> Result<(), math_explorer::pure_math::analysis::roots::AnalysisError> {
///     let f = |x: f64| x * x - 4.0;
///
///     let root_bisect = Bisection::default().find_root(f, 0.0, 5.0)?;
///     assert!((root_bisect - 2.0).abs() < math_explorer::registry::TOLERANCE_FAST);
///
///     let root_brent = Brent::default().find_root(f, 0.0, 5.0)?;
///     assert!((root_brent - 2.0).abs() < math_explorer::registry::TOLERANCE_STANDARD);
///
///     let root_newton = NewtonRaphson::default().find_root(f, 0.0, 5.0)?;
///     assert!((root_newton - 2.0).abs() < math_explorer::registry::TOLERANCE_FAST);
///
///     let root_secant = Secant::default().find_root(f, 1.0, 3.0)?;
///     assert!((root_secant - 2.0).abs() < math_explorer::registry::TOLERANCE_FAST);
///
///     Ok(())
/// }
/// ```
pub trait RootFinder {
    /// Finds a root of the function `f`.
    ///
    /// The interpretation of `min` and `max` depends on the implementation:
    /// * **Bracketing Methods** ([`Bisection`], [`Brent`]): The root must be bracketed within `[min, max]`.
    /// * **Open Methods** ([`NewtonRaphson`], [`Secant`]): `min` and `max` seed initial iterate(s),
    ///   and iterates are not constrained to `[min, max]`.
    #[verified_engine::verified]
    fn find_root<F>(&self, f: F, min: f64, max: f64) -> Result<f64, AnalysisError>
    where
        F: Fn(f64) -> f64;
}

// [cite:graph_parameters_rust]
// [cite:advanced_linear_algebra]
