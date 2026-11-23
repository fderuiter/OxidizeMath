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

/// Errors that can occur during numerical root finding and analysis.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum AnalysisError {
    /// The algorithm failed to converge within the maximum number of iterations.
    /// Contains the best guess (or final iterate / residual summary) so far.
    #[error("Algorithm failed to converge. Best guess: {0}")]
    ConvergenceError(f64),
    /// Invalid parameters were provided (e.g., root not bracketed, `NaN`/`Inf` evaluation, or singular Jacobian).
    #[error("Invalid parameters: {0}")]
    InvalidParameters(String),
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
///     assert!((root_brent - 2.0).abs() < math_explorer::registry::TOLERANCE_FAST);
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
