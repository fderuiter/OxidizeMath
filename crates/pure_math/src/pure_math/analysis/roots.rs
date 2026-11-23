//! Root finding algorithms.
//!
//! This module implements the **Strategy Pattern** for root finding, allowing
//! different algorithms (Bisection, Newton-Raphson, Brent's) to be used interchangeably.

use thiserror::Error;

/// Errors that can occur during numerical analysis.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum AnalysisError {
    /// The algorithm failed to converge within the maximum number of iterations.
    /// Contains the best guess so far.
    #[error("Algorithm failed to converge. Best guess: {0}")]
    ConvergenceError(f64),
    /// Invalid parameters were provided (e.g., root not bracketed).
    #[error("Invalid parameters: {0}")]
    InvalidParameters(String),
}

/// Strategy for finding roots of a function $f(x) = 0$.
///
/// # Examples
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::analysis::roots::{RootFinder, Bisection, NewtonRaphson};
///
/// fn main() -> Result<(), math_explorer::pure_math::analysis::roots::AnalysisError> {
///     // Define the function f(x) = x^2 - 4
///     let f = |x: f64| x * x - 4.0;
///
///     // Use Bisection (bracketing method)
///     let bisection = Bisection::default();
///     let root = bisection.find_root(f, 0.0, 5.0)?;
///     assert!((root - 2.0).abs() < math_explorer::registry::TOLERANCE_FAST);
///
///     // Use Newton-Raphson (open method)
///     let newton = NewtonRaphson::default();
///     // Note: For open methods, min/max are used to determine the initial guess.
///     let root = newton.find_root(f, 0.0, 5.0)?;
///     assert!((root - 2.0).abs() < math_explorer::registry::TOLERANCE_FAST);
///
///     Ok(())
/// }
/// ```
pub trait RootFinder {
    /// Finds a root of the function `f`.
    ///
    /// The interpretation of `min` and `max` depends on the implementation:
    /// * **Bracketing Methods** (e.g., [`Bisection`]): The root must lie within `[min, max]`.
    /// * **Open Methods** (e.g., [`NewtonRaphson`]): `min` and `max` are used to compute
    ///   the initial guess (typically `(min + max) / 2`), but the search is not constrained
    ///   to this interval.
    ///
    /// # Arguments
    /// * `f` - The objective function.
    /// * `min` - Lower bound (or parameter for initial guess).
    /// * `max` - Upper bound (or parameter for initial guess).
    ///
    /// # Returns
    /// * `Ok(f64)` - The approximate root.
    /// * `Err(AnalysisError)` - If the root cannot be found or parameters are invalid.
    #[verified_engine::verified]
    fn find_root<F>(&self, f: F, min: f64, max: f64) -> Result<f64, AnalysisError>
    where
        F: Fn(f64) -> f64;
}

/// Bisection method implementation.
///
/// A robust but slow method that iteratively halves the interval.
/// Guaranteed to converge if the function is continuous and changes sign over the interval.
///
/// # Examples
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::analysis::roots::{RootFinder, Bisection};
///
/// fn main() -> Result<(), math_explorer::pure_math::analysis::roots::AnalysisError> {
///     let solver = Bisection::default();
///
///     // Find root of x^2 - 2 = 0 in [1, 2]
///     let root = solver.find_root(|x| x * x - 2.0, 1.0, 2.0)?;
///     assert!((root - std::f64::consts::SQRT_2).abs() < math_explorer::registry::TOLERANCE_FAST);
///
///     // Error if root is not bracketed (signs at endpoints must differ)
///     let result = solver.find_root(|x| x * x + 1.0, -2.0, 2.0);
///     assert!(result.is_err());
///
///     Ok(())
/// }
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Bisection {
    /// Maximum number of iterations before giving up.
    pub max_iterations: usize,
    /// Absolute tolerance for the root value (convergence criteria).
    pub tolerance: f64,
}

impl Bisection {
    /// Creates a new Bisection solver.
    #[verified_engine::verified]
    pub fn new(max_iterations: usize, tolerance: f64) -> Self {
        Self {
            max_iterations,
            tolerance,
        }
    }
}

impl Default for Bisection {
    #[verified_engine::verified]
    fn default() -> Self {
        Self {
            max_iterations: math_commons::registry::MAX_ITERATIONS,
            tolerance: math_commons::registry::TOLERANCE_FAST,
        }
    }
}

impl RootFinder for Bisection {
    #[verified_engine::verified]
    fn find_root<F>(&self, f: F, min: f64, max: f64) -> Result<f64, AnalysisError>
    where
        F: Fn(f64) -> f64,
    {
        let mut low = min;
        let mut high = max;

        let mut f_low = f(low);
        if f_low.abs() < self.tolerance {
            return Ok(low);
        }

        let f_high = f(high);
        if f_high.abs() < self.tolerance {
            return Ok(high);
        }

        // Check if root is bracketed
        if f_low.signum() == f_high.signum() {
            return Err(AnalysisError::InvalidParameters(format!(
                "Root not bracketed: f({})={}, f({})={}. Signs must differ.",
                low, f_low, high, f_high
            )));
        }

        let mut mid = low;

        for _ in 0..self.max_iterations {
            mid = (low + high) / 2.0;

            // Check convergence on domain (interval size)
            if (high - low).abs() < self.tolerance {
                return Ok(mid);
            }

            let f_mid = f(mid);

            // Check convergence on range (function value)
            if f_mid.abs() < self.tolerance {
                return Ok(mid);
            }

            // Narrow the interval
            // If f(low) and f(mid) have different signs, root is in [low, mid]
            if f_low.signum() != f_mid.signum() {
                high = mid;
                // high moved, low stays same, so f_low is still valid for next iter
            } else {
                low = mid;
                f_low = f_mid; // low moved, must update f_low
            }
        }

        // Return best guess (mid) if convergence not reached
        Err(AnalysisError::ConvergenceError(mid))
    }
}

/// Newton-Raphson method implementation.
///
/// Uses the derivative of the function to converge quadratically to the root.
/// If the derivative is not provided (via `RootFinder` trait), a numerical approximation is used.
///
/// **Note:** This is an [Open Method](https://en.wikipedia.org/wiki/Root-finding_algorithms#Open_methods).
/// Unlike bracketing methods (like [`Bisection`]), it does not require the root to be bracketed
/// and does not guarantee that the result lies within the initial interval provided to [`RootFinder::find_root`].
/// The `min` and `max` arguments are only used to compute the initial guess: `(min + max) / 2`.
///
/// # Examples
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::analysis::roots::{RootFinder, NewtonRaphson};
///
/// fn main() -> Result<(), math_explorer::pure_math::analysis::roots::AnalysisError> {
///     let solver = NewtonRaphson::default();
///
///     // Find root of x^2 - 2 = 0
///     // Initial guess will be (1.0 + 2.0) / 2.0 = 1.5
///     let root = solver.find_root(|x| x * x - 2.0, 1.0, 2.0)?;
///     assert!((root - std::f64::consts::SQRT_2).abs() < math_explorer::registry::TOLERANCE_FAST);
///
///     // Example where root is outside the initial "interval"
///     // Root of x - 10 = 0 is 10. Initial guess is 2.5.
///     let root = solver.find_root(|x| x - 10.0, 0.0, 5.0)?;
///     assert!((root - 10.0).abs() < math_explorer::registry::TOLERANCE_FAST);
///
///     Ok(())
/// }
/// ```
#[derive(Debug, Clone, Copy)]
pub struct NewtonRaphson {
    /// Maximum number of iterations.
    pub max_iterations: usize,
    /// Absolute tolerance for the root value.
    pub tolerance: f64,
}

impl NewtonRaphson {
    /// Creates a new Newton-Raphson solver.
    #[verified_engine::verified]
    pub fn new(max_iterations: usize, tolerance: f64) -> Self {
        Self {
            max_iterations,
            tolerance,
        }
    }

    /// Finds a root using an analytical derivative $f'(x)$.
    ///
    /// # Arguments
    /// * `f` - The objective function.
    /// * `f_prime` - The derivative of the objective function.
    /// * `guess` - Initial guess for the root.
    #[verified_engine::verified]
    pub fn find_root_with_derivative<F, D>(
        &self,
        f: F,
        f_prime: D,
        guess: f64,
    ) -> Result<f64, AnalysisError>
    where
        F: Fn(f64) -> f64,
        D: Fn(f64) -> f64,
    {
        let mut x = guess;

        for _ in 0..self.max_iterations {
            let y = f(x);
            if y.abs() < self.tolerance {
                return Ok(x);
            }

            let dy = f_prime(x);
            if dy.abs() < 1e-14 {
                return Err(AnalysisError::InvalidParameters(format!(
                    "Derivative too small at x={}: {}",
                    x, dy
                )));
            }

            let next_x = x - y / dy;

            // Check convergence on domain step
            if (next_x - x).abs() < self.tolerance {
                return Ok(next_x);
            }

            x = next_x;
        }

        Err(AnalysisError::ConvergenceError(x))
    }
}

impl Default for NewtonRaphson {
    #[verified_engine::verified]
    fn default() -> Self {
        Self {
            max_iterations: math_commons::registry::MAX_ITERATIONS,
            tolerance: math_commons::registry::TOLERANCE_FAST,
        }
    }
}

impl RootFinder for NewtonRaphson {
    #[verified_engine::verified]
    fn find_root<F>(&self, f: F, min: f64, max: f64) -> Result<f64, AnalysisError>
    where
        F: Fn(f64) -> f64,
    {
        // Use midpoint as initial guess
        let guess = (min + max) / 2.0;

        // Numerical derivative step size
        let h = 1e-7;

        let derivative = |x: f64| (f(x + h) - f(x - h)) / (2.0 * h);

        // We use the specialized method with our numerical derivative
        self.find_root_with_derivative(&f, derivative, guess)
    }
}

/// Strategy for derivative-free 1D scalar minimization on a bounded interval `[min, max]`.
pub trait OneDimMinimizer {
    /// Finds a local minimizer `(x_min, f_min)` of `f` over `[min, max]`.
    ///
    /// # Errors
    /// Returns [`AnalysisError::InvalidParameters`] if the interval is invalid or non-finite,
    /// or [`AnalysisError::ConvergenceError`] if convergence is not reached within `max_iterations`.
    #[verified_engine::verified]
    fn minimize<F>(&self, f: F, min: f64, max: f64) -> Result<(f64, f64), AnalysisError>
    where
        F: FnMut(f64) -> f64;
}

/// Brent's derivative-free 1D scalar minimizer combining golden-section search
/// with successive parabolic interpolation.
///
/// # Examples
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::analysis::roots::{BrentMinimizer, OneDimMinimizer};
///
/// let solver = BrentMinimizer::default();
/// let (x_min, f_min) = solver.minimize(|x| (x - 1.75) * (x - 1.75) + 3.0, -2.0, 5.0).unwrap();
/// assert!((x_min - 1.75).abs() < math_explorer::registry::TOLERANCE_FAST);
/// assert!((f_min - 3.0).abs() < math_explorer::registry::TOLERANCE_FAST);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BrentMinimizer {
    /// Maximum number of iterations before returning a convergence error.
    pub max_iterations: usize,
    /// Relative/absolute interval tolerance for convergence.
    pub tolerance: f64,
}

impl BrentMinimizer {
    /// Creates a new Brent 1D minimizer.
    #[verified_engine::verified]
    pub fn new(max_iterations: usize, tolerance: f64) -> Self {
        Self {
            max_iterations,
            tolerance,
        }
    }
}

impl Default for BrentMinimizer {
    #[verified_engine::verified]
    fn default() -> Self {
        Self {
            max_iterations: math_commons::registry::MAX_ITERATIONS,
            tolerance: math_commons::registry::TOLERANCE_FAST,
        }
    }
}

impl OneDimMinimizer for BrentMinimizer {
    #[verified_engine::verified]
    fn minimize<F>(&self, f: F, min: f64, max: f64) -> Result<(f64, f64), AnalysisError>
    where
        F: FnMut(f64) -> f64,
    {
        if !min.is_finite() || !max.is_finite() || min >= max || self.tolerance <= 0.0 {
            return Err(AnalysisError::InvalidParameters(
                "BrentMinimizer requires finite bounds with min < max and tolerance > 0".into(),
            ));
        }
        let rel_tol = self.tolerance.sqrt().min(1e-7);
        super::optimization::derivative_free::brent_minimize_core(
            f,
            min,
            max,
            self.max_iterations,
            rel_tol,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[verified_engine::verified]
    fn test_newton_raphson_sqrt() -> Result<(), AnalysisError> {
        let solver = NewtonRaphson::default();
        // x^2 - 2 = 0, derivative 2x
        let root = solver.find_root_with_derivative(|x| x * x - 2.0, |x| 2.0 * x, 1.5)?;
        assert!((root - std::f64::consts::SQRT_2).abs() < math_commons::registry::TOLERANCE_FAST);
        Ok(())
    }

    #[test]
    #[verified_engine::verified]
    fn test_newton_raphson_numerical() -> Result<(), AnalysisError> {
        let solver = NewtonRaphson::default();
        // x^2 - 2 = 0
        let root = solver.find_root(|x| x * x - 2.0, 1.0, 2.0)?;
        assert!((root - std::f64::consts::SQRT_2).abs() < 1e-5);
        Ok(())
    }

    #[test]
    #[verified_engine::verified]
    fn test_bisection_square_root() -> Result<(), AnalysisError> {
        let solver = Bisection::default();
        // x^2 - 2 = 0  => x = sqrt(2) approx 1.41421356
        let root = solver.find_root(|x| x * x - 2.0, 1.0, 2.0)?;
        assert!((root - std::f64::consts::SQRT_2).abs() < math_commons::registry::TOLERANCE_FAST);
        Ok(())
    }

    #[test]
    #[verified_engine::verified]
    fn test_bisection_linear() -> Result<(), AnalysisError> {
        let solver = Bisection::default();
        // 2x - 4 = 0 => x = 2
        let root = solver.find_root(|x| 2.0 * x - 4.0, 0.0, 5.0)?;
        assert!((root - 2.0).abs() < math_commons::registry::TOLERANCE_FAST);
        Ok(())
    }

    #[test]
    #[verified_engine::verified]
    fn test_not_bracketed() {
        let solver = Bisection::default();
        // x^2 + 1 = 0 has no real roots. And signs are always positive.
        let result = solver.find_root(|x| x * x + 1.0, -2.0, 2.0);
        assert!(matches!(result, Err(AnalysisError::InvalidParameters(_))));
    }

    #[test]
    #[verified_engine::verified]
    fn test_brent_minimizer_quartic_and_trig() -> Result<(), AnalysisError> {
        let solver = BrentMinimizer::default();
        let (x_q, f_q) =
            solver.minimize(|x| (x - 2.25).powi(2) + 0.1 * (x - 2.25).powi(4), -3.0, 6.0)?;
        assert!((x_q - 2.25).abs() < 1e-5);
        assert!(f_q.abs() < 1e-10);

        let (x_c, f_c) = solver.minimize(f64::cos, 2.0, 4.5)?;
        assert!((x_c - std::f64::consts::PI).abs() < math_commons::registry::TOLERANCE_FAST);
        assert!((f_c + 1.0).abs() < math_commons::registry::TOLERANCE_FAST);
        Ok(())
    }
}
