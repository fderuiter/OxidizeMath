//! Root finding algorithms.
//!
//! This module implements the **Strategy Pattern** for root finding, allowing
//! different algorithms ([`Bisection`], [`NewtonRaphson`], [`Brent`], [`IntervalNewton`],
//! and [`Krawczyk`]) to be used interchangeably.

#[path = "enclosure_roots.rs"]
mod enclosure_roots;

pub use enclosure_roots::{IntervalNewton, Krawczyk};

use math_scalars::Dual;
use thiserror::Error;

/// Copy, zero-allocation errors that can occur during numerical root finding.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum AnalysisError {
    /// The algorithm failed to converge within the maximum number of iterations.
    /// Contains the best guess so far.
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
    /// Derivative magnitude fell below the safe division threshold in [`NewtonRaphson`].
    #[error("Derivative too small at x={x}: {derivative}")]
    DerivativeTooSmall {
        /// Iterate where the derivative vanished.
        x: f64,
        /// Evaluated derivative value.
        derivative: f64,
    },
    /// Non-finite interval bounds or non-positive solver tolerance were supplied.
    #[error("Invalid solver bracket [{min}, {max}] or tolerance")]
    InvalidBracket {
        /// Lower bound input.
        min: f64,
        /// Upper bound input.
        max: f64,
    },
}

/// Strategy for finding roots of a function $f(x) = 0$.
///
/// # Examples
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::analysis::roots::{Bisection, Brent, NewtonRaphson, RootFinder};
///
/// fn main() -> Result<(), math_explorer::pure_math::analysis::roots::AnalysisError> {
///     let f = |x: f64| x * x - 4.0;
///
///     let bisection = Bisection::default();
///     let root = bisection.find_root(f, 0.0, 5.0)?;
///     assert!((root - 2.0).abs() < math_explorer::registry::TOLERANCE_FAST);
///
///     let brent = Brent::default();
///     let root = brent.find_root(f, 0.0, 5.0)?;
///     assert!((root - 2.0).abs() < math_explorer::registry::TOLERANCE_STANDARD);
///
///     let newton = NewtonRaphson::default();
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
    /// * **Bracketing Methods** ([`Bisection`], [`Brent`]): The root must lie within `[min, max]`.
    /// * **Open Methods** ([`NewtonRaphson`]): `min` and `max` are used to compute
    ///   the initial guess `(min + max) / 2`, without constraining iterates to `[min, max]`.
    #[verified_engine::verified]
    fn find_root<F>(&self, f: F, min: f64, max: f64) -> Result<f64, AnalysisError>
    where
        F: Fn(f64) -> f64;
}

#[verified_engine::verified]
fn validate_solver_bounds(min: f64, max: f64, tolerance: f64) -> Result<(), AnalysisError> {
    if !min.is_finite() || !max.is_finite() || !tolerance.is_finite() || tolerance <= 0.0 {
        return Err(AnalysisError::InvalidBracket { min, max });
    }
    Ok(())
}

/// Bisection method implementation.
///
/// Iteratively halves a sign-changing interval `[min, max]`.
///
/// # Examples
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::analysis::roots::{Bisection, RootFinder};
///
/// fn main() -> Result<(), math_explorer::pure_math::analysis::roots::AnalysisError> {
///     let solver = Bisection::default();
///     let root = solver.find_root(|x| x * x - 2.0, 1.0, 2.0)?;
///     assert!((root - std::f64::consts::SQRT_2).abs() < math_explorer::registry::TOLERANCE_FAST);
///
///     let result = solver.find_root(|x| x * x + 1.0, -2.0, 2.0);
///     assert!(result.is_err());
///     Ok(())
/// }
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
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
        validate_solver_bounds(min, max, self.tolerance)?;
        let mut low = min;
        let mut high = max;
        let mut f_low = f(low);
        let f_high = f(high);
        if !f_low.is_finite() || !f_high.is_finite() {
            return Err(AnalysisError::RootNotBracketed {
                min: low,
                f_min: f_low,
                max: high,
                f_max: f_high,
            });
        }
        if f_low.abs() < self.tolerance {
            return Ok(low);
        }
        if f_high.abs() < self.tolerance {
            return Ok(high);
        }
        if (f_low > 0.0) == (f_high > 0.0) {
            return Err(AnalysisError::RootNotBracketed {
                min: low,
                f_min: f_low,
                max: high,
                f_max: f_high,
            });
        }

        let mut mid = low;
        for _ in 0..self.max_iterations {
            mid = 0.5 * (low + high);
            if (high - low).abs() < self.tolerance {
                return Ok(mid);
            }
            let f_mid = f(mid);
            if !f_mid.is_finite() {
                return Err(AnalysisError::ConvergenceError(mid));
            }
            if f_mid.abs() < self.tolerance {
                return Ok(mid);
            }
            if (f_low > 0.0) != (f_mid > 0.0) {
                high = mid;
            } else {
                low = mid;
                f_low = f_mid;
            }
        }
        Err(AnalysisError::ConvergenceError(mid))
    }
}

/// Newton-Raphson open root-finding implementation.
///
/// Uses analytical or central finite-difference derivatives starting from `(min + max) / 2`.
///
/// # Examples
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::analysis::roots::{NewtonRaphson, RootFinder};
///
/// fn main() -> Result<(), math_explorer::pure_math::analysis::roots::AnalysisError> {
///     let solver = NewtonRaphson::default();
///     let root = solver.find_root(|x| x * x - 2.0, 1.0, 2.0)?;
///     assert!((root - std::f64::consts::SQRT_2).abs() < math_explorer::registry::TOLERANCE_FAST);
///     Ok(())
/// }
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
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
        validate_solver_bounds(guess, guess, self.tolerance)?;
        let mut x = guess;

        for _ in 0..self.max_iterations {
            let y = f(x);
            if !y.is_finite() {
                return Err(AnalysisError::ConvergenceError(x));
            }
            if y.abs() < self.tolerance {
                return Ok(x);
            }

            let dy = f_prime(x);
            if !dy.is_finite() || dy.abs() < 1e-14 {
                return Err(AnalysisError::DerivativeTooSmall { x, derivative: dy });
            }

            let next_x = x - y / dy;
            if !next_x.is_finite() {
                return Err(AnalysisError::ConvergenceError(x));
            }
            let step_tol = self.tolerance.max(4.0 * f64::EPSILON * next_x.abs());
            if (next_x - x).abs() <= step_tol {
                return Ok(next_x);
            }
            x = next_x;
        }

        Err(AnalysisError::ConvergenceError(x))
    }

    /// Finds a root using exact zero-allocation forward-mode [`Dual<f64>`] automatic differentiation.
    ///
    /// Reference: Griewank & Walther (2008), *Evaluating Derivatives*, SIAM, DOI: 10.1137/1.9780898717761.
    ///
    /// # Stability
    /// Eliminates finite-difference step underflow (`x + h == x`) at large magnitudes such as
    /// `|x| >= 3.0e9`, converging quadratically up to scale-aware ULP resolution.
    ///
    /// # Domain of Validity
    /// Requires finite `guess`, positive `self.tolerance`, and non-vanishing derivative $|f'(x_k)| \ge 10^{-14}$.
    #[verified_engine::verified]
    pub fn find_root_ad<F>(&self, f: F, guess: f64) -> Result<f64, AnalysisError>
    where
        F: Fn(Dual<f64>) -> Dual<f64>,
    {
        validate_solver_bounds(guess, guess, self.tolerance)?;
        let mut x = guess;

        for _ in 0..self.max_iterations {
            let jet = f(Dual::variable(x));
            if !jet.val.is_finite() {
                return Err(AnalysisError::ConvergenceError(x));
            }
            if jet.val.abs() < self.tolerance {
                return Ok(x);
            }
            if !jet.eps.is_finite() || jet.eps.abs() < 1e-14 {
                return Err(AnalysisError::DerivativeTooSmall {
                    x,
                    derivative: jet.eps,
                });
            }
            let next_x = x - jet.val / jet.eps;
            if !next_x.is_finite() {
                return Err(AnalysisError::ConvergenceError(x));
            }
            let step_tol = self.tolerance.max(4.0 * f64::EPSILON * next_x.abs());
            if (next_x - x).abs() <= step_tol {
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
        validate_solver_bounds(min, max, self.tolerance)?;
        let guess = 0.5 * (min + max);
        let derivative = |x: f64| {
            let h = f64::EPSILON.sqrt() * x.abs().max(1.0);
            (f(x + h) - f(x - h)) / (2.0 * h)
        };
        self.find_root_with_derivative(&f, derivative, guess)
    }
}

/// Bracketed Brent-Dekker root-finding solver.
///
/// Combines inverse quadratic interpolation, secant steps, and guaranteed bisection fallback
/// for superlinear convergence without derivative evaluations or dynamic allocation.
///
/// # Examples
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::analysis::roots::{Brent, RootFinder};
///
/// fn main() -> Result<(), math_explorer::pure_math::analysis::roots::AnalysisError> {
///     let solver = Brent::default();
///     let root = solver.find_root(|x| x.cos() - x, 0.0, 1.0)?;
///     assert!((root - 0.739_085_133_215).abs() < math_explorer::registry::TOLERANCE_STANDARD);
///     Ok(())
/// }
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Brent {
    /// Maximum number of iterations before returning [`AnalysisError::ConvergenceError`].
    pub max_iterations: usize,
    /// Absolute bracket convergence tolerance.
    pub tolerance: f64,
}

impl Brent {
    /// Creates a new [`Brent`] solver with explicit iteration and tolerance bounds.
    #[verified_engine::verified]
    pub fn new(max_iterations: usize, tolerance: f64) -> Self {
        Self {
            max_iterations,
            tolerance,
        }
    }
}

impl Default for Brent {
    #[verified_engine::verified]
    fn default() -> Self {
        Self {
            max_iterations: math_commons::registry::MAX_ITERATIONS,
            tolerance: math_commons::registry::TOLERANCE_HIGH,
        }
    }
}

#[path = "brent_step.rs"]
mod brent_step;
use brent_step::{BrentIterate, brent_propose_step};

impl RootFinder for Brent {
    #[verified_engine::verified]
    fn find_root<F>(&self, f: F, min: f64, max: f64) -> Result<f64, AnalysisError>
    where
        F: Fn(f64) -> f64,
    {
        validate_solver_bounds(min, max, self.tolerance)?;
        let fa0 = f(min);
        let fb0 = f(max);
        if !fa0.is_finite() || !fb0.is_finite() {
            return Err(AnalysisError::RootNotBracketed {
                min,
                f_min: fa0,
                max,
                f_max: fb0,
            });
        }
        if fa0 == 0.0 {
            return Ok(min);
        }
        if fb0 == 0.0 {
            return Ok(max);
        }
        if (fa0 > 0.0) == (fb0 > 0.0) {
            return Err(AnalysisError::RootNotBracketed {
                min,
                f_min: fa0,
                max,
                f_max: fb0,
            });
        }

        let mut st = BrentIterate {
            a: min,
            b: max,
            c: max,
            fa: fa0,
            fb: fb0,
            fc: fb0,
            d: max - min,
            e: max - min,
        };

        for _ in 0..self.max_iterations {
            if (st.fb > 0.0) == (st.fc > 0.0) {
                st.c = st.a;
                st.fc = st.fa;
                st.d = st.b - st.a;
                st.e = st.d;
            }
            if st.fc.abs() < st.fb.abs() {
                st.a = st.b;
                st.b = st.c;
                st.c = st.a;
                st.fa = st.fb;
                st.fb = st.fc;
                st.fc = st.fa;
            }
            let tol1 = 2.0 * f64::EPSILON * st.b.abs() + 0.5 * self.tolerance;
            let xm = 0.5 * (st.c - st.b);
            if xm.abs() <= tol1 || st.fb == 0.0 {
                return Ok(st.b);
            }
            let (next_d, next_e) = brent_propose_step(&st, xm, tol1);
            st.d = next_d;
            st.e = next_e;
            st.a = st.b;
            st.fa = st.fb;
            if st.d.abs() > tol1 {
                st.b += st.d;
            } else {
                st.b += tol1.copysign(xm);
            }
            st.fb = f(st.b);
            if !st.fb.is_finite() {
                return Err(AnalysisError::ConvergenceError(st.b));
            }
        }

        Err(AnalysisError::ConvergenceError(st.b))
    }
}

#[cfg(test)]
#[path = "roots_tests.rs"]
mod tests;

// [cite:advanced_linear_algebra]
