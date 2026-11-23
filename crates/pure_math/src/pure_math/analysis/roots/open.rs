//! Open scalar root-finding methods: [`NewtonRaphson`] and [`Secant`].

use super::{AnalysisError, RootFinder};
use crate::pure_math::analysis::dense_linear::canonical_zero;

/// Newton-Raphson open root-finding method.
///
/// Uses analytical or central finite-difference derivatives to converge quadratically near
/// simple roots. When invoked through [`RootFinder::find_root`], `(min + max) / 2` serves as
/// the initial guess.
///
/// # Examples
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::analysis::roots::{NewtonRaphson, RootFinder};
///
/// let solver = NewtonRaphson::default();
/// let root = solver.find_root(|x| x * x - 2.0, 1.0, 2.0).unwrap();
/// assert!((root - std::f64::consts::SQRT_2).abs() < math_explorer::registry::TOLERANCE_FAST);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NewtonRaphson {
    /// Maximum number of iterations.
    pub max_iterations: usize,
    /// Absolute tolerance for the root and residual value.
    pub tolerance: f64,
}

impl NewtonRaphson {
    /// Creates a new [`NewtonRaphson`] solver.
    #[verified_engine::verified]
    pub fn new(max_iterations: usize, tolerance: f64) -> Self {
        Self {
            max_iterations,
            tolerance,
        }
    }

    /// Finds a root using an analytical derivative $f'(x)$ starting from `guess`.
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
        if !guess.is_finite() || !self.tolerance.is_finite() || self.tolerance <= 0.0 {
            return Err(AnalysisError::InvalidBracket {
                min: guess,
                max: guess,
            });
        }

        let mut x = guess;
        for _ in 0..self.max_iterations {
            let y = f(x);
            if !y.is_finite() {
                return Err(AnalysisError::ConvergenceError(canonical_zero(x)));
            }
            if y == 0.0 || y.abs() < self.tolerance {
                return Ok(canonical_zero(x));
            }

            let dy = f_prime(x);
            if !dy.is_finite() || dy.abs() < 1e-14 {
                return Err(AnalysisError::DerivativeTooSmall { x, derivative: dy });
            }

            let next_x = x - y / dy;
            if !next_x.is_finite() {
                return Err(AnalysisError::ConvergenceError(canonical_zero(x)));
            }
            if (next_x - x).abs() < self.tolerance {
                return Ok(canonical_zero(next_x));
            }
            x = next_x;
        }

        Err(AnalysisError::ConvergenceError(canonical_zero(x)))
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
        if !min.is_finite()
            || !max.is_finite()
            || !self.tolerance.is_finite()
            || self.tolerance <= 0.0
        {
            return Err(AnalysisError::InvalidBracket { min, max });
        }
        let guess = min + 0.5 * (max - min);
        let h = 1e-7;
        let derivative = |x: f64| (f(x + h) - f(x - h)) / (2.0 * h);
        self.find_root_with_derivative(&f, derivative, guess)
    }
}

/// Secant two-point open root-finding method.
///
/// Approximates derivatives via successive secant chords without requiring analytical
/// or finite-difference derivative evaluations, achieving order $\varphi \approx 1.618$
/// superlinear convergence on smooth roots.
///
/// # Examples
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::analysis::roots::{RootFinder, Secant};
///
/// let solver = Secant::default();
/// let root = solver.find_root(|x| x.exp() - 3.0 * x, 0.0, 1.0).unwrap();
/// assert!((root - 0.619_061_286).abs() < math_explorer::registry::TOLERANCE_FAST);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Secant {
    /// Maximum number of iterations before returning [`AnalysisError::ConvergenceError`].
    pub max_iterations: usize,
    /// Absolute convergence tolerance.
    pub tolerance: f64,
}

impl Secant {
    /// Creates a new [`Secant`] solver.
    #[verified_engine::verified]
    pub fn new(max_iterations: usize, tolerance: f64) -> Self {
        Self {
            max_iterations,
            tolerance,
        }
    }

    /// Finds a root starting from explicit initial iterates `(x0, x1)`.
    #[verified_engine::verified]
    pub fn find_root_from_pair<F>(&self, f: F, x0: f64, x1: f64) -> Result<f64, AnalysisError>
    where
        F: Fn(f64) -> f64,
    {
        if !x0.is_finite() || !x1.is_finite() || !self.tolerance.is_finite() || self.tolerance < 0.0
        {
            return Err(AnalysisError::InvalidBracket { min: x0, max: x1 });
        }

        let mut prev_x = x0;
        let mut curr_x = if (x1 - x0).abs() < 1e-12 {
            x0 + 1e-4 * x0.abs().max(1.0)
        } else {
            x1
        };

        let mut prev_f = f(prev_x);
        if !prev_f.is_finite() {
            return Err(AnalysisError::InvalidParameters(
                "Non-finite evaluation at initial Secant point x0",
            ));
        }
        if prev_f == 0.0 || prev_f.abs() <= self.tolerance {
            return Ok(canonical_zero(prev_x));
        }

        let mut curr_f = f(curr_x);
        if !curr_f.is_finite() {
            return Err(AnalysisError::InvalidParameters(
                "Non-finite evaluation at initial Secant point x1",
            ));
        }
        if curr_f == 0.0 || curr_f.abs() <= self.tolerance {
            return Ok(canonical_zero(curr_x));
        }

        for _ in 0..self.max_iterations {
            let denom = curr_f - prev_f;
            if denom.abs() < 1e-14 {
                return Err(AnalysisError::DerivativeTooSmall {
                    x: curr_x,
                    derivative: denom,
                });
            }
            let next_x = curr_x - curr_f * (curr_x - prev_x) / denom;
            if !next_x.is_finite() {
                return Err(AnalysisError::ConvergenceError(canonical_zero(curr_x)));
            }
            let next_f = f(next_x);
            if !next_f.is_finite() {
                return Err(AnalysisError::InvalidParameters(
                    "Non-finite evaluation during Secant step",
                ));
            }
            if next_f == 0.0
                || next_f.abs() <= self.tolerance
                || (next_x - curr_x).abs() <= self.tolerance
            {
                return Ok(canonical_zero(next_x));
            }
            prev_x = curr_x;
            prev_f = curr_f;
            curr_x = next_x;
            curr_f = next_f;
        }

        Err(AnalysisError::ConvergenceError(canonical_zero(curr_x)))
    }
}

impl Default for Secant {
    #[verified_engine::verified]
    fn default() -> Self {
        Self {
            max_iterations: math_commons::registry::MAX_ITERATIONS,
            tolerance: math_commons::registry::TOLERANCE_FAST,
        }
    }
}

impl RootFinder for Secant {
    #[verified_engine::verified]
    fn find_root<F>(&self, f: F, min: f64, max: f64) -> Result<f64, AnalysisError>
    where
        F: Fn(f64) -> f64,
    {
        self.find_root_from_pair(f, min, max)
    }
}

// [cite:graph_parameters_rust]
